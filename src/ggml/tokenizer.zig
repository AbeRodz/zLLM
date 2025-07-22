const std = @import("std");
const json = std.json;

pub const TokenEntry = struct {
    piece: []const u8,
    score: f32,
    type: u32,
    is_unknown: bool,
    is_control: bool,
    is_unused: bool,
    is_byte: bool,
};

pub const TokenizerData = struct {
    tokens: []TokenEntry,
    special_tokens: std.StringHashMap(usize),
    add_special_tokens: std.StringHashMap(bool),
    chat_template: []const u8,
};

pub fn parseTokenizerJson(allocator: std.mem.Allocator, path: []const u8) !TokenizerData {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = (try file.stat()).size;
    const buffer = try allocator.alloc(u8, file_size);
    errdefer allocator.free(buffer);

    _ = try file.readAll(buffer);

    const parsed = try json.parseFromSlice(json.Value, allocator, buffer, .{
        .allocate = .alloc_always, // So you don't keep referencing the original buffer
    });
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidJson;

    const root_obj = root.object;

    const tokens_obj = root_obj.get("tokens") orelse return error.MissingTokensField;
    if (tokens_obj != .object) return error.InvalidTokens;

    const special_tokens_obj = root_obj.get("special_tokens");

    var tokens = std.ArrayList(TokenEntry).init(allocator);
    var special_tokens = std.StringHashMap(usize).init(allocator);

    // Parse "tokens"
    var iter = tokens_obj.object.iterator();
    while (iter.next()) |entry| {
        _ = entry.key_ptr.*;
        const token_data = entry.value_ptr.*;

        if (token_data != .object) continue;
        const token_obj = token_data.object;

        const piece = token_obj.get("piece") orelse continue;
        const score = token_obj.get("score") orelse continue;
        const token_type = token_obj.get("type") orelse continue;
        const is_unknown = token_obj.get("is_unknown") orelse continue;
        const is_control = token_obj.get("is_control") orelse continue;
        const is_unused = token_obj.get("is_unused") orelse continue;
        const is_byte = token_obj.get("is_byte") orelse continue;

        try tokens.append(TokenEntry{
            .piece = piece.string,
            .score = @as(f32, @floatCast(score.float)),
            .type = @as(u32, @intCast(token_type.integer)),
            .is_unknown = is_unknown.bool,
            .is_control = is_control.bool,
            .is_unused = is_unused.bool,
            .is_byte = is_byte.bool,
        });
    }

    // Parse "special_tokens" if present
    if (special_tokens_obj) |st_obj| {
        if (st_obj == .object) {
            var st_iter = st_obj.object.iterator();
            while (st_iter.next()) |entry| {
                const name = entry.key_ptr.*;
                const id = entry.value_ptr.*;
                if (id == .integer) {
                    try special_tokens.put(name, @as(usize, @intCast(id.integer)));
                }
            }
        }
    }
    const add_special_tokens_obj = root_obj.get("add_special_tokens");

    var add_special_tokens = std.StringHashMap(bool).init(allocator);
    if (add_special_tokens_obj) |obj| {
        if (obj == .object) {
            var add_st_iter = obj.object.iterator();
            while (add_st_iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const val = entry.value_ptr.*;
                if (val == .bool) {
                    try add_special_tokens.put(key, val.bool);
                }
            }
        }
    }
    const chat_template_obj = root_obj.get("chat_template");

    var chat_template: []const u8 = "";
    if (chat_template_obj) |obj| {
        if (obj == .string) {
            chat_template = std.mem.sliceTo(obj.string, 0);
        }
    }

    return TokenizerData{
        .tokens = try tokens.toOwnedSlice(),
        .special_tokens = special_tokens,
        .add_special_tokens = add_special_tokens,
        .chat_template = chat_template,
    };
}
