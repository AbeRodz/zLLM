// custom parser for handling cases within the openAI standard, cannot use tokamak's dynamic validator.
const std = @import("std");
const completions = @import("models/completions.zig");
const Content = completions.Content;
const ContentPart = completions.ContentPart;
const TextContentPart = completions.TextContentPart;
const ImageContentPart = completions.ImageContentPart;
const AudioContentPart = completions.AudioContentPart;
const FileContentPart = completions.FileContentPart;
const Message = completions.Message;
const Tool = completions.Tool;
const ToolCall = completions.ToolCall;
const ResponseFormat = completions.ResponseFormat;
const JsonSchema = completions.JsonSchema;
const StreamOptions = completions.StreamOptions;
const ContentObject = completions.ContentObject;
const ChatCompletionRequest = completions.ChatCompletionRequest;

// ---------------------------------------------------------------------------
// JSON value → string serializer (for parameters and arguments fields)
// ---------------------------------------------------------------------------

// In Zig 0.15, ArrayList mutation methods require an explicit allocator argument.
fn writeJsonValue(allocator: std.mem.Allocator, list: *std.ArrayList(u8), value: std.json.Value) error{OutOfMemory}!void {
    switch (value) {
        .null => try list.appendSlice(allocator, "null"),
        .bool => |b| try list.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |n| {
            var buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{n}) catch "0";
            try list.appendSlice(allocator, s);
        },
        .float => |f| {
            var buf: [64]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{f}) catch "0.0";
            try list.appendSlice(allocator, s);
        },
        .number_string => |s| try list.appendSlice(allocator, s),
        .string => |s| {
            try list.append(allocator, '"');
            for (s) |c| {
                switch (c) {
                    '"' => try list.appendSlice(allocator, "\\\""),
                    '\\' => try list.appendSlice(allocator, "\\\\"),
                    '\n' => try list.appendSlice(allocator, "\\n"),  // 0x0a
                    '\r' => try list.appendSlice(allocator, "\\r"),  // 0x0d
                    '\t' => try list.appendSlice(allocator, "\\t"),  // 0x09
                    // remaining C0 control chars (excluding \t, \n, \r already handled above)
                    0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => {
                        var esc: [6]u8 = undefined;
                        const hex = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch "\\u0000";
                        try list.appendSlice(allocator, hex);
                    },
                    else => try list.append(allocator, c),
                }
            }
            try list.append(allocator, '"');
        },
        .array => |arr| {
            try list.append(allocator, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try list.append(allocator, ',');
                try writeJsonValue(allocator, list, item);
            }
            try list.append(allocator, ']');
        },
        .object => |obj| {
            try list.append(allocator, '{');
            var it = obj.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try list.append(allocator, ',');
                first = false;
                try list.append(allocator, '"');
                try list.appendSlice(allocator, entry.key_ptr.*);
                try list.appendSlice(allocator, "\":");
                try writeJsonValue(allocator, list, entry.value_ptr.*);
            }
            try list.append(allocator, '}');
        },
    }
}

fn jsonValueToString(allocator: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    var buf = try std.ArrayList(u8).initCapacity(allocator, 256);
    try writeJsonValue(allocator, &buf, value);
    return buf.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Content parts
// ---------------------------------------------------------------------------

pub fn decodeParts(allocator: std.mem.Allocator, value: std.json.Value) ![]ContentPart {
    if (value != .array) return error.ExpectedArray;
    const array = value.array;

    var parts = try allocator.alloc(ContentPart, array.items.len);

    for (array.items, 0..) |item, i| {
        if (item != .object) return error.ExpectedObject;
        const obj = item.object;

        const type_val = obj.get("type") orelse return error.MissingField;
        if (type_val != .string) return error.ExpectedString;
        const type_str = type_val.string;

        if (std.mem.eql(u8, type_str, "text")) {
            const text_val = obj.get("text") orelse return error.MissingField;
            if (text_val != .string) return error.ExpectedString;

            parts[i] = ContentPart{
                .text = TextContentPart{
                    .type = .text,
                    .text = text_val.string,
                },
            };
        } else if (std.mem.eql(u8, type_str, "image")) {
            parts[i] = ContentPart{ .image = ImageContentPart{ .type = .image } };
        } else if (std.mem.eql(u8, type_str, "audio")) {
            parts[i] = ContentPart{ .audio = AudioContentPart{ .type = .audio } };
        } else if (std.mem.eql(u8, type_str, "file")) {
            parts[i] = ContentPart{ .file = FileContentPart{ .type = .file } };
        } else {
            return error.UnknownType;
        }
    }

    return parts;
}

fn decodeContent(allocator: std.mem.Allocator, value: std.json.Value) !Content {
    return switch (value) {
        // null content — assistant messages that only carry tool_calls
        .null => Content{ .none = {} },
        .string => Content{ .plain = value.string },
        // content: [{"type":"text","text":"..."}] — array-of-parts form used by
        // newer OpenAI SDKs, CopilotKit, and vision-capable clients.
        .array => Content{ .object = ContentObject{ .parts = try decodeParts(allocator, value) } },
        .object => blk: {
            const obj = value.object;

            if (obj.get("text")) |text_val| {
                if (text_val != .string) return error.InvalidContent;
                break :blk Content{ .object = ContentObject{ .text = text_val.string } };
            }

            if (obj.get("parts")) |parts_val| {
                break :blk Content{ .object = ContentObject{ .parts = try decodeParts(allocator, parts_val) } };
            }

            return error.InvalidContent;
        },
        else => error.InvalidContent,
    };
}

// ---------------------------------------------------------------------------
// Tool calls (in conversation history assistant messages)
// ---------------------------------------------------------------------------

fn decodeToolCall(allocator: std.mem.Allocator, value: std.json.Value) !ToolCall {
    if (value != .object) return error.ExpectedObject;
    const obj = value.object;

    const id_val = obj.get("id") orelse return error.MissingField;
    if (id_val != .string) return error.ExpectedString;

    const type_str: []const u8 = if (obj.get("type")) |tv|
        if (tv == .string) tv.string else "function"
    else
        "function";

    const func_val = obj.get("function") orelse return error.MissingField;
    if (func_val != .object) return error.ExpectedObject;
    const func_obj = func_val.object;

    const name_val = func_obj.get("name") orelse return error.MissingField;
    if (name_val != .string) return error.ExpectedString;

    const args_val = func_obj.get("arguments") orelse std.json.Value{ .string = "{}" };
    const arguments: []const u8 = switch (args_val) {
        .string => |s| s,
        // Some clients send arguments as a direct object instead of a JSON string.
        else => try jsonValueToString(allocator, args_val),
    };

    return ToolCall{
        .id = id_val.string,
        .index = 0,
        .type = type_str,
        .function = .{
            .name = name_val.string,
            .arguments = arguments,
        },
    };
}

fn decodeToolCalls(allocator: std.mem.Allocator, value: std.json.Value) ![]ToolCall {
    if (value != .array) return error.ExpectedArray;
    const arr = value.array;
    const result = try allocator.alloc(ToolCall, arr.items.len);
    for (arr.items, 0..) |item, i| {
        result[i] = try decodeToolCall(allocator, item);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Tools (request-side function definitions)
// ---------------------------------------------------------------------------

fn decodeTool(allocator: std.mem.Allocator, value: std.json.Value) !Tool {
    if (value != .object) return error.ExpectedObject;
    const obj = value.object;

    const type_str: []const u8 = if (obj.get("type")) |tv|
        if (tv == .string) tv.string else "function"
    else
        "function";

    const func_val = obj.get("function") orelse return error.MissingField;
    if (func_val != .object) return error.ExpectedObject;
    const func_obj = func_val.object;

    const name_val = func_obj.get("name") orelse return error.MissingField;
    if (name_val != .string) return error.ExpectedString;

    const description: ?[]const u8 = if (func_obj.get("description")) |dv|
        if (dv == .string) dv.string else null
    else
        null;

    // Serialize the parameters schema back to a JSON string so we can
    // embed it verbatim in the tool system prompt.
    const parameters: ?[]const u8 = if (func_obj.get("parameters")) |pv|
        switch (pv) {
            .null => null,
            else => try jsonValueToString(allocator, pv),
        }
    else
        null;

    return Tool{
        .type = type_str,
        .function = .{
            .name = name_val.string,
            .description = description,
            .parameters = parameters,
        },
    };
}

fn decodeTools(allocator: std.mem.Allocator, value: std.json.Value) ![]Tool {
    if (value != .array) return error.ExpectedArray;
    const arr = value.array;
    const result = try allocator.alloc(Tool, arr.items.len);
    for (arr.items, 0..) |item, i| {
        result[i] = try decodeTool(allocator, item);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Stream options / stop / response_format
// ---------------------------------------------------------------------------

fn decodeStreamOptions(value: std.json.Value) !?StreamOptions {
    return switch (value) {
        .object => blk: {
            const obj = value.object;

            if (obj.get("include_usage")) |bool_val| {
                if (bool_val != .bool) return error.InvalidContent;
                break :blk StreamOptions{ .include_usage = bool_val.bool };
            }

            return error.InvalidContent;
        },
        else => error.InvalidContent,
    };
}

fn decodeStop(val: std.json.Value, allocator: std.mem.Allocator) !?[]?[]const u8 {
    switch (val) {
        // Single string stop sequence — wrap in a one-element slice.
        .string => |s| {
            const result = try allocator.alloc(?[]const u8, 1);
            result[0] = s;
            return result;
        },
        // Array of stop sequences (strings or null sentinels).
        .array => |array| {
            const result = try allocator.alloc(?[]const u8, array.items.len);
            for (array.items, 0..) |item, i| {
                result[i] = switch (item) {
                    .null => null,
                    .string => item.string,
                    else => return error.ExpectedStringOrNull,
                };
            }
            return result;
        },
        else => return error.ExpectedStringOrArray,
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Decode messages from an already-parsed JSON root value.
/// String slices borrow from `root`; caller must keep `root` alive.
fn decodeMessagesFromValue(allocator: std.mem.Allocator, root: std.json.Value) ![]Message {
    const messages_json = root.object.get("messages").?;
    if (messages_json != .array) return error.InvalidMessages;

    const messages = try allocator.alloc(Message, messages_json.array.items.len);
    for (messages_json.array.items, 0..) |msg_val, i| {
        const msg_obj = msg_val.object;
        const role = msg_obj.get("role").?.string;

        // content may be absent or null for tool-call assistant messages
        const content_raw = msg_obj.get("content") orelse std.json.Value{ .null = {} };
        const content = try decodeContent(allocator, content_raw);

        const tool_calls: ?[]ToolCall = if (msg_obj.get("tool_calls")) |tc_val|
            switch (tc_val) {
                .null => null,
                .array => try decodeToolCalls(allocator, tc_val),
                else => null,
            }
        else
            null;

        const tool_call_id: ?[]const u8 = if (msg_obj.get("tool_call_id")) |id_val|
            if (id_val == .string) id_val.string else null
        else
            null;

        messages[i] = Message{
            .role = role,
            .content = content,
            .tool_calls = tool_calls,
            .tool_call_id = tool_call_id,
        };
    }
    return messages;
}

pub fn decodeMessages(allocator: std.mem.Allocator, body: []const u8) ![]Message {
    // Note: parsed is intentionally NOT deinit'd here because the returned
    // Message slices borrow string data from parsed.value.  The arena that
    // owns `allocator` is responsible for cleanup at request end.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    return decodeMessagesFromValue(allocator, parsed.value);
}

// ---------------------------------------------------------------------------
// response_format
// ---------------------------------------------------------------------------

fn decodeResponseFormat(val: std.json.Value, allocator: std.mem.Allocator) !ResponseFormat {
    if (val != .object) return error.ExpectedObject;

    const obj = val.object;

    const type_val = obj.get("type") orelse return error.MissingField;
    const type_str = try std.json.parseFromValue([]const u8, allocator, type_val, .{});

    const schema_val = obj.get("json_schema");
    const json_schema = if (schema_val) |v| blk: {
        if (v == .null) break :blk null;
        const schema_obj = v.object;
        const schema_field = schema_obj.get("schema") orelse return error.MissingSchemaField;
        const schema_str = try std.json.parseFromValue([]const u8, allocator, schema_field, .{});
        break :blk JsonSchema{ .schema = schema_str.value };
    } else null;

    return ResponseFormat{
        .type = type_str.value,
        .json_schema = json_schema,
    };
}

// ---------------------------------------------------------------------------
// Generic field helper
// ---------------------------------------------------------------------------

fn getFieldAs(comptime T: type, value: std.json.Value, key: []const u8, allocator: std.mem.Allocator) !?T {
    const val = value.object.get(key) orelse return null;
    const parsed = try std.json.parseFromValue(T, allocator, val, .{});
    return parsed.value;
}

// ---------------------------------------------------------------------------
// Top-level request decoder
// ---------------------------------------------------------------------------

pub fn decodeCompletionRequest(allocator: std.mem.Allocator, body: []const u8) !ChatCompletionRequest {
    // Note: parsed is intentionally NOT deinit'd — all string slices in the
    // returned ChatCompletionRequest borrow from parsed.value.  The arena
    // that owns `allocator` frees everything at request end.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});

    const root = parsed.value;
    const stream_opt = root.object.get("stream_options");
    const response_fmt = root.object.get("response_format");
    const stop_val = root.object.get("stop");
    const tools_val = root.object.get("tools");
    const model = try getFieldAs([]const u8, root, "model", allocator);
    // Reuse the already-parsed root to avoid a second full JSON parse.
    const messages = try decodeMessagesFromValue(allocator, root);
    const stream = try getFieldAs(bool, root, "stream", allocator);
    const max_tokens = try getFieldAs(i32, root, "max_tokens", allocator);
    const seed = try getFieldAs(i32, root, "seed", allocator);
    const stop = if (stop_val) |val| try decodeStop(val, allocator) else null;
    const stream_options = if (stream_opt) |val| try decodeStreamOptions(val) else null;
    const temperature = try getFieldAs(f64, root, "temperature", allocator);
    const frequency_penalty = try getFieldAs(f64, root, "frequency_penalty", allocator);
    const presence_penalty = try getFieldAs(f64, root, "presence_penalty", allocator);
    const top_p = try getFieldAs(f64, root, "top_p", allocator);
    const response_format = if (response_fmt) |val| try decodeResponseFormat(val, allocator) else null;
    const tools = if (tools_val) |val| switch (val) {
        .null => null,
        .array => try decodeTools(allocator, val),
        else => null,
    } else null;
    const tool_choice = try getFieldAs([]const u8, root, "tool_choice", allocator);

    return ChatCompletionRequest{
        .model = model.?,
        .messages = messages,
        .stream = stream,
        .stream_options = stream_options,
        .max_tokens = max_tokens,
        .seed = seed,
        .stop = stop,
        .temperature = temperature,
        .frequency_penalty = frequency_penalty,
        .presence_penalty = presence_penalty,
        .top_p = top_p,
        .response_format = response_format,
        .tools = tools,
        .tool_choice = tool_choice,
    };
}

// ---------------------------------------------------------------------------
// Content text extraction
// ---------------------------------------------------------------------------

pub fn getTextContent(msg: Message) ?[]const u8 {
    return switch (msg.content) {
        .none => null,
        .plain => |text| if (text.len > 0) text else null,
        .object => |obj| switch (obj) {
            .text => |text| if (text.len > 0) text else null,
            else => null,
        },
    };
}
