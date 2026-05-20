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

    var tokens: std.ArrayList(TokenEntry) = .empty;
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

        try tokens.append(allocator, TokenEntry{
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
        .tokens = try tokens.toOwnedSlice(allocator),
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
    var token_list: std.ArrayList(TokenArray) = .empty;

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
        try token_list.append(allocator, .{
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
    token_list.deinit(allocator);

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

// ---------------------------------------------------------------------------
// HuggingFace BPE tokenizer (tokenizer.json + tokenizer_config.json)
// ---------------------------------------------------------------------------

pub const BPETokenizerData = struct {
    tokens: []const []const u8,
    token_types: []u32,
    merges: []const []const u8,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
    pad_token_id: u32,
    chat_template: []const u8,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *BPETokenizerData) void {
        self.arena.deinit();
    }
};

/// Extract the token string from a tokenizer_config.json special-token field.
/// The field can be a plain string or an object with a "content" key.
fn extractTokenStr(v: json.Value) []const u8 {
    return switch (v) {
        .string => v.string,
        .object => if (v.object.get("content")) |c|
            (if (c == .string) c.string else "")
        else
            "",
        else => "",
    };
}

fn lookupTokenId(tokens: []const []const u8, needle: []const u8) u32 {
    for (tokens, 0..) |tok, i| {
        if (std.mem.eql(u8, tok, needle)) return @as(u32, @intCast(i));
    }
    return 0;
}

/// Parse a HuggingFace BPE tokenizer.json + tokenizer_config.json pair.
/// All memory lives in the embedded arena; call data.deinit() to free.
pub fn parseBPETokenizerJson(
    backing_allocator: std.mem.Allocator,
    tokenizer_json_path: []const u8,
    tokenizer_config_path: []const u8,
) !BPETokenizerData {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();

    // ── tokenizer.json ────────────────────────────────────────────────────
    const tj = blk: {
        var f = try std.fs.cwd().openFile(tokenizer_json_path, .{});
        defer f.close();
        const sz = (try f.stat()).size;
        const buf = try alloc.alloc(u8, sz);
        _ = try f.readAll(buf);
        break :blk try json.parseFromSliceLeaky(json.Value, alloc, buf, .{ .allocate = .alloc_always });
    };
    if (tj != .object) return error.InvalidJson;
    const tj_obj = tj.object;

    const model_val = tj_obj.get("model") orelse return error.MissingModel;
    if (model_val != .object) return error.InvalidModel;
    const model_obj = model_val.object;

    // vocab: {"token_str": id}  →  sorted [][]const u8
    const vocab_val = model_obj.get("vocab") orelse return error.MissingVocab;
    if (vocab_val != .object) return error.InvalidVocab;

    var vocab_size: usize = 0;
    {
        var it = vocab_val.object.iterator();
        while (it.next()) |e| {
            const id = @as(usize, @intCast(e.value_ptr.*.integer));
            if (id + 1 > vocab_size) vocab_size = id + 1;
        }
    }

    const tokens = try alloc.alloc([]const u8, vocab_size);
    for (tokens) |*t| t.* = "";
    const token_types = try alloc.alloc(u32, vocab_size);
    @memset(token_types, 1); // NORMAL

    {
        var it = vocab_val.object.iterator();
        while (it.next()) |e| {
            const id = @as(usize, @intCast(e.value_ptr.*.integer));
            tokens[id] = e.key_ptr.*;
        }
    }

    // Mark added_tokens (special=true) as CONTROL (3)
    if (tj_obj.get("added_tokens")) |added_val| {
        if (added_val == .array) {
            for (added_val.array.items) |item| {
                if (item != .object) continue;
                const id_val = item.object.get("id") orelse continue;
                const spec_val = item.object.get("special") orelse continue;
                if (spec_val == .bool and spec_val.bool) {
                    const id = @as(usize, @intCast(id_val.integer));
                    if (id < vocab_size) token_types[id] = 3; // CONTROL
                }
            }
        }
    }

    // merges: ["Ġ t", ...]
    const merges_val = model_obj.get("merges") orelse return error.MissingMerges;
    if (merges_val != .array) return error.InvalidMerges;
    const merges = try alloc.alloc([]const u8, merges_val.array.items.len);
    for (merges_val.array.items, 0..) |m, i| {
        merges[i] = if (m == .string) m.string else "";
    }

    // ── tokenizer_config.json ─────────────────────────────────────────────
    const tc = blk: {
        var f = try std.fs.cwd().openFile(tokenizer_config_path, .{});
        defer f.close();
        const sz = (try f.stat()).size;
        const buf = try alloc.alloc(u8, sz);
        _ = try f.readAll(buf);
        break :blk try json.parseFromSliceLeaky(json.Value, alloc, buf, .{ .allocate = .alloc_always });
    };
    const tc_obj = if (tc == .object) tc.object else return error.InvalidTokenizerConfig;

    const bos_str = if (tc_obj.get("bos_token")) |v| extractTokenStr(v) else "";
    const eos_str = if (tc_obj.get("eos_token")) |v| extractTokenStr(v) else "";
    const unk_str = if (tc_obj.get("unk_token")) |v| extractTokenStr(v) else "";
    const pad_str = if (tc_obj.get("pad_token")) |v| extractTokenStr(v) else "";
    const chat_tmpl = if (tc_obj.get("chat_template")) |v| (if (v == .string) v.string else "") else "";

    return BPETokenizerData{
        .tokens = tokens,
        .token_types = token_types,
        .merges = merges,
        .bos_token_id = lookupTokenId(tokens, bos_str),
        .eos_token_id = lookupTokenId(tokens, eos_str),
        .unk_token_id = lookupTokenId(tokens, unk_str),
        .pad_token_id = lookupTokenId(tokens, pad_str),
        .chat_template = chat_tmpl,
        .arena = arena,
    };
}
