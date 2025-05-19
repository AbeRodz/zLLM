// custom parser for handling cases within the openAI standard, cannot use tokamak's dynamic validator.
const std = @import("std");
const Content = @import("completions.zig").Content;
const ContentPart = @import("completions.zig").ContentPart;
const TextContentPart = @import("completions.zig").TextContentPart;
const ImageContentPart = @import("completions.zig").ImageContentPart;
const AudioContentPart = @import("completions.zig").AudioContentPart;
const FileContentPart = @import("completions.zig").FileContentPart;
const Message = @import("completions.zig").Message;
const ResponseFormat = @import("completions.zig").ResponseFormat;
const JsonSchema = @import("completions.zig").JsonSchema;
const StreamOptions = @import("completions.zig").StreamOptions;
const ContentObject = @import("completions.zig").ContentObject;
const ChatCompletionRequest = @import("completions.zig").ChatCompletionRequest;

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
        .string => Content{ .plain = value.string },
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
    if (val != .array) return error.ExpectedArray;

    const array = val.array;
    const result = try allocator.alloc(?[]const u8, array.items.len);

    for (array.items, 0..) |item, i| {
        result[i] = switch (item) {
            .null => null,
            .string => item.string,
            else => return error.ExpectedStringOrNull,
        };
    }

    return result;
}

pub fn decodeMessages(allocator: std.mem.Allocator, body: []const u8) ![]Message {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const root = parsed.value;

    const messages_json = root.object.get("messages").?;

    if (messages_json != .array) return error.InvalidMessages;

    const messages = try allocator.alloc(Message, messages_json.array.items.len);
    for (messages_json.array.items, 0..) |msg_val, i| {
        const msg_obj = msg_val.object;
        const role = msg_obj.get("role").?.string;
        const content_val = msg_obj.get("content").?;

        const content = try decodeContent(allocator, content_val);

        messages[i] = Message{
            .role = role,
            .content = content,
        };
    }
    return messages;
}

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

fn getFieldAs(comptime T: type, value: std.json.Value, key: []const u8, allocator: std.mem.Allocator) !?T {
    const val = value.object.get(key) orelse return null;
    const parsed = try std.json.parseFromValue(T, allocator, val, .{});
    return parsed.value;
}

pub fn decodeCompletionRequest(allocator: std.mem.Allocator, body: []const u8) !ChatCompletionRequest {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const stream_opt = root.object.get("stream_options");
    const response_fmt = root.object.get("response_format");
    const stop_val = root.object.get("stop");
    const model = try getFieldAs([]const u8, root, "model", allocator);
    const messages = try decodeMessages(allocator, body);
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
        //.tools = tools,
    };
}
pub fn getTextContent(msg: Message) ?[]const u8 {
    return switch (msg.content) {
        .plain => |text| if (text.len > 0) text else null,
        .object => |obj| switch (obj) {
            .text => |text| if (text.len > 0) text else null,
            else => null,
        },
    };
}

// pub fn filterSystemTags(messages: []Message) []Message {
//     for (messages) |msg| {
//         if (std.mem.eql(u8, msg.role,"system") {

//         };
//     }
// }
