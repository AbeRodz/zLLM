const std = @import("std");
const tk = @import("tokamak");
const completions = @import("models/completions.zig");
const ChatCompletionRequest = completions.ChatCompletionRequest;
const ChatCompletionResponse = completions.ChatCompletionResponse;
const ChatCompletionChunk = completions.ChatCompletionChunk;
const Choice = completions.Choice;
const ResponseMessage = completions.ResponseMessage;
const Content = completions.Content;
const Tool = completions.Tool;
const ToolCall = completions.ToolCall;
const llama = @import("../llama/llama.zig");
const chat_format = llama.chat_format;
const CompletionStreamer = @import("completions_stream.zig").CompletionStreamer;
const parser = @import("completions_parser.zig");
const errh = @import("error_handler.zig");
const uuid = @import("uuid");

// ---------------------------------------------------------------------------
// SSE streaming — runs in a dedicated thread per stream
// ---------------------------------------------------------------------------

const StreamCtx = struct {
    arena: *std.heap.ArenaAllocator,
    streamer: *CompletionStreamer,
};

/// Called by tokamak in a dedicated thread with the raw TCP stream.
/// Owns the backpressure slot: acquireSlot() is called in the handler,
/// releaseSlot() is called here in the defer block.
fn runSSE(cx: StreamCtx, stream: std.net.Stream) void {
    const server_alloc = cx.arena.child_allocator;
    defer {
        cx.streamer.deinit();
        server_alloc.destroy(cx.streamer);
        stream.close();
        cx.arena.deinit();
        server_alloc.destroy(cx.arena);
        errh.releaseSlot(); // paired with acquireSlot() in handleCompletionStream
    }

    while (cx.streamer.next()) |maybe| {
        const chunk = maybe orelse break;
        sendSSEChunk(stream, chunk) catch {
            // Client disconnected — stop the decode loop immediately so we
            // don't burn GPU cycles on tokens nobody will receive.
            cx.streamer.cancel();
            break;
        };
    } else |e| {
        sendSSEError(stream, @errorName(e)) catch {};
    }

    // OpenAI SSE terminator — Python SDK / CopilotKit hang without this.
    sendSSERaw(stream, "data: [DONE]\n\n") catch {};
}

fn sendSSEChunk(stream: std.net.Stream, chunk: ChatCompletionChunk) !void {
    // 1 KiB stack buffer: a typical chunk is ~250 bytes, the stop chunk ~400.
    // All writes from JSON stringify accumulate here, then flush() emits a
    // single sendmsg() syscall instead of ~20 per token.
    var buf: [1024]u8 = undefined;
    var sw = stream.writer(&buf);
    const writer = &sw.interface;
    try writer.writeAll("data: ");
    // emit_null_optional_fields=false omits null role/content from delta objects,
    // matching the OpenAI wire format (subsequent deltas have no "role" key).
    try std.json.Stringify.value(chunk, .{ .emit_null_optional_fields = false }, writer);
    try writer.writeAll("\n\n");
    try writer.flush();
}

fn sendSSEError(stream: std.net.Stream, msg: []const u8) !void {
    var buf: [512]u8 = undefined;
    var sw = stream.writer(&buf);
    const writer = &sw.interface;
    try writer.writeAll("data: ");
    try std.json.Stringify.value(.{ .@"error" = msg }, .{}, writer);
    try writer.writeAll("\n\n");
    try writer.flush();
}

fn sendSSERaw(stream: std.net.Stream, data: []const u8) !void {
    // Single small write — buffer it and flush in one syscall.
    var buf: [64]u8 = undefined;
    var sw = stream.writer(&buf);
    try sw.interface.writeAll(data);
    try sw.interface.flush();
}

// ---------------------------------------------------------------------------
// Tool system prompt builder
// ---------------------------------------------------------------------------

fn appendJsonEscaped(allocator: std.mem.Allocator, list: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            else => try list.append(allocator, c),
        }
    }
}

/// Build a system prompt that instructs the model about available tools.
///
/// Uses the <tool_call> format that Gemma function-calling models expect and
/// that our lazy GBNF grammar sampler is triggered by.  Other instruction-tuned
/// models that follow prompt instructions will also use this format when asked.
fn buildToolSystemMessage(allocator: std.mem.Allocator, tools: []const Tool) ![]const u8 {
    var buf = std.ArrayList(u8){};

    try buf.appendSlice(allocator,
        "You have access to the following tools.\n\n" ++
        "When you need to call a tool, respond with exactly this format " ++
        "(no text before or after the tag):\n" ++
        "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>\n\n" ++
        "After receiving a [Tool Result ...] message, provide your answer in plain " ++
        "text — do NOT call the tool again.\n\n" ++
        "Available tools:\n",
    );

    for (tools) |tool| {
        try buf.appendSlice(allocator, "- ");
        try appendJsonEscaped(allocator, &buf, tool.function.name);
        if (tool.function.description) |desc| {
            try buf.appendSlice(allocator, ": ");
            try appendJsonEscaped(allocator, &buf, desc);
        }
        if (tool.function.parameters) |params| {
            try buf.appendSlice(allocator, "\n  Parameters: ");
            try buf.appendSlice(allocator, params);
        }
        try buf.append(allocator, '\n');
    }

    return buf.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Tool conversion — API layer → llama layer
// ---------------------------------------------------------------------------

/// Convert a slice of API-layer Tool definitions to the minimal ApiTool type
/// that the llama layer understands.  Slices borrow from the original Tool
/// structs; caller must keep those alive.
fn toApiTools(
    allocator: std.mem.Allocator,
    tools: []const Tool,
) ![]const chat_format.ApiTool {
    const result = try allocator.alloc(chat_format.ApiTool, tools.len);
    for (tools, 0..) |t, i| {
        result[i] = .{
            .name = t.function.name,
            .description = t.function.description,
            .parameters_json = t.function.parameters,
        };
    }
    return result;
}

/// Convert a chat_format.ParsedToolCall into a single-element ToolCall slice
/// suitable for the OpenAI response.
fn parsedToToolCalls(
    allocator: std.mem.Allocator,
    ptc: chat_format.ParsedToolCall,
) ![]ToolCall {
    const tcs = try allocator.alloc(ToolCall, 1);
    tcs[0] = ToolCall{
        .id = ptc.call_id,
        .index = 0,
        .type = "function",
        .function = .{
            .name = ptc.name,
            .arguments = ptc.arguments,
        },
    };
    return tcs;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Format a slice of tool calls as a compact JSON object (first call only,
/// since parallel tool calls in history are rare and hard to round-trip through
/// the text-only chat template).
fn formatToolCallsAsJson(allocator: std.mem.Allocator, tool_calls: []const ToolCall) ![]const u8 {
    if (tool_calls.len == 0) return try allocator.dupe(u8, "{}");
    const tc = tool_calls[0];
    return std.fmt.allocPrint(
        allocator,
        "{{\"name\":\"{s}\",\"arguments\":{s}}}",
        .{ tc.function.name, tc.function.arguments },
    );
}

/// Convert the API message list into the flat role+content pairs that
/// llama_chat_apply_template expects.
///
/// Tool handling:
///   • If tools are provided and no system message exists in req_messages,
///     a synthetic system message with tool definitions is prepended.
///   • If a system message already exists, tool definitions are appended to it.
///   • Messages with role "tool" are converted to user turns so they survive
///     the C-API chat template (which only knows user/assistant/system roles).
///   • Assistant messages with null content but tool_calls are serialised to
///     their JSON form so the model sees what it previously requested.
fn buildApiMessages(
    allocator: std.mem.Allocator,
    req_messages: []const completions.Message,
    tools: ?[]const Tool,
) ![]llama.ApiMessage {
    // Build the tool system prompt once if needed.
    const tool_prompt: ?[]const u8 = if (tools) |ts| blk: {
        if (ts.len == 0) break :blk null;
        break :blk try buildToolSystemMessage(allocator, ts);
    } else null;

    var list = std.ArrayListUnmanaged(llama.ApiMessage){};

    // If there is no system message in the request and we have a tool prompt,
    // prepend it as a synthetic system message.
    if (tool_prompt != null) {
        const has_system = for (req_messages) |m| {
            if (std.mem.eql(u8, m.role, "system")) break true;
        } else false;

        if (!has_system) {
            try list.append(allocator, .{ .role = "system", .content = tool_prompt.? });
        }
    }

    for (req_messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            const base = parser.getTextContent(msg) orelse "";
            const content: []const u8 = if (tool_prompt) |tp| blk: {
                if (base.len > 0) {
                    break :blk try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ base, tp });
                } else {
                    break :blk tp;
                }
            } else base;
            if (content.len > 0) {
                try list.append(allocator, .{ .role = "system", .content = content });
            }
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            // Tool results are injected as user turns so the C chat template
            // can handle them.  The tool_call_id is included for traceability.
            const result_text = parser.getTextContent(msg) orelse "";
            const formatted = if (msg.tool_call_id) |id|
                try std.fmt.allocPrint(allocator, "[Tool Result for {s}]: {s}", .{ id, result_text })
            else
                try std.fmt.allocPrint(allocator, "[Tool Result]: {s}", .{result_text});
            try list.append(allocator, .{ .role = "user", .content = formatted });
        } else if (std.mem.eql(u8, msg.role, "assistant")) {
            const content: ?[]const u8 = switch (msg.content) {
                // No text — format the tool calls the model previously requested.
                .none => if (msg.tool_calls) |tcs|
                    try formatToolCallsAsJson(allocator, tcs)
                else
                    null,
                // Regular text content (ignore any attached tool_calls here since
                // the model already expressed them as text in this message).
                else => parser.getTextContent(msg),
            };
            if (content) |c| {
                try list.append(allocator, .{ .role = "assistant", .content = c });
            }
        } else {
            // user and any other roles
            const content = parser.getTextContent(msg) orelse continue;
            try list.append(allocator, .{ .role = msg.role, .content = content });
        }
    }
    return list.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// /v2/chat/completions — debug endpoint, echoes body.
pub fn handleCompletions(ctx: tk.Context) ![]const u8 {
    const body = ctx.req.body().?;
    var parsed = try std.json.parseFromSlice(std.json.Value, ctx.allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value;
    const model = root.object.get("model").?.string;
    std.debug.print("{s}", .{model});
    std.debug.print("{s}", .{body});
    const messages = try parser.decodeMessages(ctx.allocator, body);
    for (messages) |msg| {
        const content = parser.getTextContent(msg).?;
        std.debug.print("{s}", .{content});
    }
    return "hello";
}

/// /v3/chat/completions — non-streaming only, with OpenAI error format.
pub fn handleCompletion(ctx: *tk.Context, allocator: std.mem.Allocator) !void {
    handleCompletionInner(ctx, allocator) catch |err| {
        errh.send(ctx, err);
    };
}

fn handleCompletionInner(ctx: *tk.Context, allocator: std.mem.Allocator) !void {
    errh.acquireSlot() catch |err| {
        return err;
    };
    defer errh.releaseSlot();

    const body = ctx.req.body().?;
    const req = try parser.decodeCompletionRequest(allocator, body);
    const messages = try buildApiMessages(allocator, req.messages, req.tools);
    const session_id: ?[]const u8 = ctx.req.header("X-Session-ID");
    const api_tools: ?[]const chat_format.ApiTool = if (req.tools) |ts|
        try toApiTools(allocator, ts)
    else
        null;
    const result = try llama.respondToPrompt(allocator, req.model, 8192, messages, session_id, api_tools);

    // Tool call detection is handled inside respondToPrompt via the GBNF lazy
    // grammar sampler + chat_format.parseOutput.  result.tool_call is non-null
    // when the model emitted a <tool_call> JSON block.
    const tool_calls: ?[]ToolCall = if (result.tool_call) |ptc|
        try parsedToToolCalls(allocator, ptc)
    else
        null;

    const finish_reason: []const u8 = if (tool_calls != null) "tool_calls" else "stop";
    const response_content: ?[]const u8 = if (tool_calls != null) null else result.content;

    var choices = try allocator.alloc(Choice, 1);
    choices[0] = .{
        .message = ResponseMessage{
            .content = response_content,
            .role = "assistant",
            .tool_calls = tool_calls,
        },
        .index = 0,
        .finish_reason = finish_reason,
    };
    const uuidUrn = uuid.urn.serialize(uuid.v4.new());
    const id = try std.fmt.allocPrint(allocator, "chatcmpl-{s}", .{&uuidUrn});

    const res = ChatCompletionResponse{
        .id = id,
        .object = "chat.completion",
        .model = req.model,
        .created = std.time.timestamp(),
        .system_fingerprint = "zLLM",
        .choices = choices,
        .usage = .{
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .total_tokens = result.prompt_tokens + result.completion_tokens,
        },
    };
    ctx.res.content_type = .JSON;
    ctx.res.status = 200;
    try ctx.res.json(res, .{ .emit_null_optional_fields = false });
    try ctx.res.write();
}

/// /v1/chat/completions — streaming and non-streaming, with OpenAI error format.
pub fn handleCompletionStream(ctx: *tk.Context, allocator: std.mem.Allocator) !void {
    handleCompletionStreamInner(ctx, allocator) catch |err| {
        errh.send(ctx, err);
    };
}

fn handleCompletionStreamInner(ctx: *tk.Context, allocator: std.mem.Allocator) !void {
    // Reject immediately if at capacity or shutting down — before any GPU work.
    try errh.acquireSlot();
    // Note: for the streaming path the slot is released inside runSSE (the
    // handler returns before streaming finishes).  For the non-streaming path
    // the defer below owns the release.
    var streaming = false;
    defer if (!streaming) errh.releaseSlot();

    const body = ctx.req.body().?;
    const req = try parser.decodeCompletionRequest(allocator, body);
    const messages = try buildApiMessages(allocator, req.messages, req.tools);
    const api_tools: ?[]const chat_format.ApiTool = if (req.tools) |ts|
        try toApiTools(allocator, ts)
    else
        null;

    // The Go gateway injects X-Session-ID for sticky-routed sessions.
    // When absent (direct calls, health checks) we fall back to stateless.
    const session_id: ?[]const u8 = ctx.req.header("X-Session-ID");

    if (req.stream == true) {
        // Objects must outlive this handler frame — use the server allocator.
        const server_alloc = ctx.server.allocator;

        const arena = try server_alloc.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(server_alloc);
        errdefer {
            arena.deinit();
            server_alloc.destroy(arena);
        }

        const streamer = try server_alloc.create(CompletionStreamer);
        errdefer server_alloc.destroy(streamer);

        streamer.* = try CompletionStreamer.init(arena.allocator(), req, messages, session_id, api_tools);
        errdefer streamer.deinit();

        const cx = StreamCtx{ .arena = arena, .streamer = streamer };
        // startEventStream spawns the runSSE thread.  From this point the
        // slot is owned by runSSE — prevent the defer above from releasing it.
        try ctx.res.startEventStream(cx, runSSE);
        streaming = true;
        return;
    }

    // Non-streaming path — defer releases the slot.
    const result = try llama.respondToPrompt(allocator, req.model, 8192, messages, session_id, api_tools);

    const tool_calls: ?[]ToolCall = if (result.tool_call) |ptc|
        try parsedToToolCalls(allocator, ptc)
    else
        null;

    const finish_reason: []const u8 = if (tool_calls != null) "tool_calls" else "stop";
    const response_content: ?[]const u8 = if (tool_calls != null) null else result.content;

    var choices = try allocator.alloc(Choice, 1);
    choices[0] = .{
        .message = ResponseMessage{
            .content = response_content,
            .role = "assistant",
            .tool_calls = tool_calls,
        },
        .index = 0,
        .finish_reason = finish_reason,
    };
    const uuidUrn = uuid.urn.serialize(uuid.v4.new());
    const id = try std.fmt.allocPrint(allocator, "chatcmpl-{s}", .{&uuidUrn});

    const res = ChatCompletionResponse{
        .id = id,
        .object = "chat.completion",
        .model = req.model,
        .created = std.time.timestamp(),
        .system_fingerprint = "zLLM",
        .choices = choices,
        .usage = .{
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .total_tokens = result.prompt_tokens + result.completion_tokens,
        },
    };
    ctx.res.content_type = .JSON;
    ctx.res.status = 200;
    try ctx.res.json(res, .{ .emit_null_optional_fields = false });
    try ctx.res.write();
}

// ---------------------------------------------------------------------------
// Health endpoint
// ---------------------------------------------------------------------------

pub const HealthResponse = struct {
    status: []const u8,
    active_requests: u32,
    max_concurrent: u32,
};

pub fn handleHealth(_: tk.Context) !HealthResponse {
    return .{
        .status = if (errh.isShuttingDown()) "shutting_down" else "ok",
        .active_requests = errh.activeRequests(),
        .max_concurrent = errh.MAX_CONCURRENT,
    };
}
