const std = @import("std");
const json = std.json;

pub const SentencePieceTokenType = enum(u32) {
    NORMAL = 1,
    UNKNOWN = 2,
    CONTROL = 3,
    USER_DEFINED = 4,
    UNUSED = 5,
    BYTE = 6,
};
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
    special_tokens: std.StringArrayHashMap(usize),
    add_special_tokens: std.StringArrayHashMap(bool),
    chat_template: []const u8,
};

pub const TokenArray = struct {
    id: usize,
    entry: TokenEntry,
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
    var special_tokens = std.StringArrayHashMap(usize).init(allocator);

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

    var add_special_tokens = std.StringArrayHashMap(bool).init(allocator);
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

pub fn parseTokenizerJsonV2(allocator: std.mem.Allocator, path: []const u8) !TokenizerData {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = (try file.stat()).size;
    const buffer = try allocator.alloc(u8, file_size);
    errdefer allocator.free(buffer);

    _ = try file.readAll(buffer);

    const parsed = try json.parseFromSlice(json.Value, allocator, buffer, .{
        .allocate = .alloc_always,
    });
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidJson;
    const root_obj = root.object;

    const tokens_obj = root_obj.get("tokens") orelse return error.MissingTokensField;
    if (tokens_obj != .object) return error.InvalidTokens;

    const special_tokens_obj = root_obj.get("special_tokens");

    // --- Collect tokens with their IDs ---
    var token_list = std.ArrayList(TokenArray).init(allocator);

    var iter = tokens_obj.object.iterator();
    while (iter.next()) |entry| {
        const id_str = entry.key_ptr.*;
        const id = try std.fmt.parseInt(usize, id_str, 10);
        const token_data = entry.value_ptr.*;

        if (token_data != .object) continue;
        const token_obj = token_data.object;

        const piece = token_obj.get("piece") orelse continue;
        const score = token_obj.get("score") orelse continue;
        //const token_type = token_obj.get("type") orelse continue;
        const is_unknown = token_obj.get("is_unknown") orelse continue;
        const is_control = token_obj.get("is_control") orelse continue;
        const is_unused = token_obj.get("is_unused") orelse continue;
        const is_byte = token_obj.get("is_byte") orelse continue;
        // std.debug.print("piece:{s} score:{d} type:{d} unk:{} control:{} unused:{} byte:{}\n", .{
        //     piece.string,
        //     score.float,
        //     token_type.integer,
        //     is_unknown.bool,
        //     is_control.bool,
        //     is_unused.bool,
        //     is_byte.bool,
        // });

        var tok_type = SentencePieceTokenType.NORMAL;
        if (is_unknown.bool) {
            tok_type = SentencePieceTokenType.UNKNOWN;
        }
        if (is_control.bool) {
            tok_type = SentencePieceTokenType.CONTROL;
        }
        if (is_unused.bool) {
            tok_type = SentencePieceTokenType.UNUSED;
        }
        if (is_byte.bool) {
            tok_type = SentencePieceTokenType.BYTE;
        }
        try token_list.append(.{
            .id = id,
            .entry = TokenEntry{
                .piece = piece.string,
                .score = @as(f32, @floatCast(score.float)),
                .type = @intFromEnum(tok_type),
                .is_unknown = is_unknown.bool,
                .is_control = is_control.bool,
                .is_unused = is_unused.bool,
                .is_byte = is_byte.bool,
            },
        });
    }

    // --- Sort tokens by ID ---
    std.mem.sort(
        TokenArray,
        token_list.items,
        {},
        cmpById,
    );

    // --- Convert to ordered slice ---
    var tokens = try allocator.alloc(TokenEntry, token_list.items.len);
    for (token_list.items, 0..) |item, i| {
        tokens[i] = item.entry;
    }
    token_list.deinit();

    // --- Parse special_tokens ---
    var special_tokens = std.StringArrayHashMap(usize).init(allocator);
    if (special_tokens_obj) |st_obj| {
        if (st_obj == .object) {
            var st_iter = st_obj.object.iterator();
            while (st_iter.next()) |entry| {
                const name = entry.key_ptr.*;
                const id = entry.value_ptr.*;
                std.debug.print("specialToken name:{s} id:{d}\n", .{ name, id.integer });
                if (id == .integer) {
                    try special_tokens.put(name, @as(usize, @intCast(id.integer)));
                }
            }
        }
    }

    // --- Parse add_special_tokens ---
    const add_special_tokens_obj = root_obj.get("add_special_tokens");
    var add_special_tokens = std.StringArrayHashMap(bool).init(allocator);
    if (add_special_tokens_obj) |obj| {
        if (obj == .object) {
            var add_st_iter = obj.object.iterator();
            while (add_st_iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const val = entry.value_ptr.*;
                std.debug.print("should add specialToken name:{s} id:{}\n", .{ key, val.bool });
                if (val == .bool) {
                    try add_special_tokens.put(key, val.bool);
                }
            }
        }
    }

    // --- Parse chat_template ---
    const chat_template_obj = root_obj.get("chat_template");
    var chat_template: []const u8 = "";
    if (chat_template_obj) |obj| {
        if (obj == .string) {
            chat_template = std.mem.sliceTo(obj.string, 0);
        }
    }

    return TokenizerData{
        .tokens = tokens,
        .special_tokens = special_tokens,
        .add_special_tokens = add_special_tokens,
        .chat_template = chat_template,
    };
}

fn cmpById(_: void, a: TokenArray, b: TokenArray) bool {
    if (a.id < b.id) {
        return true;
    } else {
        return false;
    }
}
