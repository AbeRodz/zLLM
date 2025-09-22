const std = @import("std");
const tk = @import("tokamak");
const ChatCompletionRequest = @import("models/completions.zig").ChatCompletionRequest;
const ChatCompletionResponse = @import("models/completions.zig").ChatCompletionResponse;
const Choice = @import("models/completions.zig").Choice;
const ResponseMessage = @import("models/completions.zig").ResponseMessage;
const Content = @import("models/completions.zig").Content;
const llama = @import("../llama/llama.zig");
const CompletionStreamer = @import("completions_stream.zig").CompletionStreamer;
const parser = @import("completions_parser.zig");
const uuid = @import("uuid");

pub fn handleCompletions(ctx: tk.Context) ![]const u8 {
    const body = ctx.req.body().?;
    // Parse raw JSON into a tree
    var parsed = try std.json.parseFromSlice(std.json.Value, ctx.allocator, body, .{});
    defer parsed.deinit();

    const root = parsed.value;

    // Extract and validate top-level fields
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
pub fn handleCompletion(ctx: tk.Context, allocator: std.mem.Allocator) !ChatCompletionResponse {
    const body = ctx.req.body().?;

    const req = try parser.decodeCompletionRequest(allocator, body);

    var model_response: []u8 = undefined;
    var prompt = std.ArrayList(u8).init(allocator);
    defer prompt.deinit();

    for (req.messages) |msg| {
        const content = parser.getTextContent(msg).?;

        const prefix = msg.role;
        try prompt.appendSlice(prefix);
        try prompt.appendSlice(content);
        try prompt.appendSlice("\n");
    }

    model_response = try llama.respondToPrompt(allocator, req.model, 8192, prompt.items);
    var choices = try allocator.alloc(Choice, 1);
    choices[0] = .{
        .message = ResponseMessage{
            .content = model_response,
            .role = "assistant",
        },
        .index = 0,
        .finish_reason = "stop",
    };
    const uuidUrn = uuid.urn.serialize(uuid.v4.new());
    const id = try allocator.alloc(u8, uuidUrn.len);
    @memcpy(id, &uuidUrn);

    return ChatCompletionResponse{
        .id = id,
        .object = "chat.completion",
        .model = req.model,
        .created = std.time.timestamp(),
        .system_fingerprint = "zLLM",
        .choices = choices,
        .usage = .{
            .prompt_tokens = 1,
            .completion_tokens = 1,
            .total_tokens = 2,
        },
    };
}

pub fn handleCompletionStream(ctx: *tk.Context, allocator: std.mem.Allocator) !void {
    const body = ctx.req.body().?;
    const req = try parser.decodeCompletionRequest(allocator, body);

    var prompt = std.ArrayList(u8).init(allocator);
    defer prompt.deinit();

    for (req.messages) |msg| {
        const content = parser.getTextContent(msg);
        if (content) |val| {
            const prefix = msg.role;
            try prompt.appendSlice(prefix);
            try prompt.appendSlice(val);
            try prompt.appendSlice("\n");
        }
    }

    if (req.stream == true) {
        const impl = try CompletionStreamer.init(allocator, req, prompt.items);
        const StreamType = tk.EventStream(CompletionStreamer);
        var stream = StreamType{ .impl = impl };
        try stream.sendResponse(ctx);
        return;
    }

    // non-streaming response
    const full_response = try llama.respondToPrompt(allocator, req.model, 8192, prompt.items);

    var choices = try allocator.alloc(Choice, 1);
    choices[0] = .{
        .message = ResponseMessage{
            .content = full_response,
            .role = "assistant",
        },
        .index = 0,
        .finish_reason = "stop",
    };
    const uuidUrn = uuid.urn.serialize(uuid.v4.new());
    const id = try allocator.alloc(u8, uuidUrn.len);
    @memcpy(id, &uuidUrn);

    const res = ChatCompletionResponse{
        .id = id,
        .object = "chat.completion",
        .model = req.model,
        .created = std.time.timestamp(),
        .system_fingerprint = "zLLM",
        .choices = choices,
        .usage = .{
            .prompt_tokens = 1,
            .completion_tokens = 1,
            .total_tokens = 2,
        },
    };
    ctx.res.content_type = .JSON;
    ctx.res.status = 200;
    try ctx.res.json(res, .{});
    try ctx.res.write();
}
