const std = @import("std");

pub const llama = @cImport({
    @cInclude("llama.h");
});
const ggml = @cImport({
    @cInclude("ggml.h");
});
const LLAMA_NGRAM_MIN = 1;
const LLAMA_NGRAM_MAX = 4;
const LLAMA_NGRAM_STATIC = 2;

const draft_min_sample_size_lax = [_]i32{ 2, 2, 1, 1 };
const draft_min_percent_lax = [_]i32{ 66, 50, 50, 50 };

const draft_min_sample_size_strict = [_]i32{ 4, 3, 2, 2 };
const draft_min_percent_strict = [_]i32{ 75, 66, 66, 66 };

const CommonNGram = struct {
    tokens: [LLAMA_NGRAM_MAX]llama.llama_token = [_]llama.llama_token{llama.LLAMA_TOKEN_NULL} ** LLAMA_NGRAM_MAX,
    const Self = @This();
    pub fn initEmpty() Self {
        return Self{};
    }

    pub fn initFromSlice(input: []const llama.llama_token, ngram_size: usize) Self {
        var out = Self{};
        var i: usize = 0;
        while (i < LLAMA_NGRAM_MAX) : (i += 1) {
            out.tokens[i] = if (i < ngram_size) input[i] else llama.LLAMA_TOKEN_NULL;
        }
        return out;
    }

    pub fn eql(self: *const Self, other: *const Self) bool {
        var i: usize = 0;
        while (i < LLAMA_NGRAM_MAX) : (i += 1) {
            if (self.tokens[i] != other.tokens[i]) return false;
        }
        return true;
    }

    pub fn commonHash(self: *const Self) u64 {
        var hash = commonTokenHash(self.tokens[0]);
        var i: usize = 1;
        while (i < LLAMA_NGRAM_MAX) : (i += 1) {
            hash ^= commonTokenHash(self.tokens[i]);
        }
        return hash;
    }
};

pub fn commonTokenHash(token: llama.llama_token) u64 {
    return @as(u64, @intCast(token)) * 11400714819323198485;
}

pub const CommonNGramContext = struct {
    allocator: *std.mem.Allocator,
    map: std.HashMap(CommonNGram, CommonNGramCachePart, CommonNGramHasher, CommonNGramComparator),

    pub fn init(allocator: *std.mem.Allocator) !CommonNGramContext {
        return .{
            .allocator = allocator,
            .map = try std.HashMap(CommonNGram, CommonNGramCachePart, CommonNGramHasher, CommonNGramComparator).init(allocator),
        };
    }
};

// token -> count
pub const CommonNGramCachePart = std.AutoHashMap(llama.llama_token, i32);

// ngram -> {token -> count}
pub const CommonNGramCache = std.HashMap(CommonNGram, CommonNGramCachePart, CommonNGramHasher, CommonNGramComparator);

pub const CommonNGramHasher = struct {
    pub fn hash(_: @TypeOf(.{}), key: CommonNGram) u64 {
        return key.commonHash();
    }
};

pub const CommonNGramComparator = struct {
    pub fn eql(_: @TypeOf(.{}), a: CommonNGram, b: CommonNGram) bool {
        return a.eql(&b);
    }
};

// Function declarations:

pub fn common_ngram_cache_update(
    allocator: std.mem.Allocator,
    ngram_cache: *CommonNGramCache,
    ngram_min: usize,
    ngram_max: usize,
    inp: []const llama.llama_token,
    nnew: usize,
    print_progress: bool,
) !void {
    const t_start_ms = ggml.ggml_time_ms();
    const inp_size = inp.len;

    const n_todo: usize = inp_size * (ngram_max - ngram_min + 1);
    var n_done: usize = 0;

    var ngram_size: usize = ngram_min;
    while (ngram_size <= ngram_max) : (ngram_size += 1) {
        const i_start = @max(inp_size - nnew, ngram_size);
        var i: usize = i_start;
        while (i < inp_size) : (i += 1) {
            const ngram_start = i - ngram_size;
            const token = inp[i];

            const ngram = CommonNGram.initFromSlice(inp[ngram_start .. ngram_start + ngram_size], ngram_size);

            const part = try ngram_cache.getOrPut(ngram);
            if (!part.found_existing) {
                var token_map = CommonNGramCachePart.init(allocator);
                try token_map.put(token, 1);
                part.value_ptr.* = token_map;
            } else {
                var token_map = part.value_ptr;
                const token_entry = try token_map.getOrPut(token);
                if (!token_entry.found_existing) {
                    token_entry.value_ptr.* = 1;
                } else {
                    token_entry.value_ptr.* += 1;
                }
            }

            n_done += 1;

            if (print_progress and n_done % 10_000_000 == 0) {
                const t_now_ms = ggml.ggml_time_ms();
                const eta_ms: usize = (n_todo - n_done) * (t_now_ms - t_start_ms) / n_done;
                const eta_min = eta_ms / (60 * 1000);
                const eta_s = (eta_ms - eta_min * 60 * 1000) / 1000;

                std.debug.print("common_ngram_cache_update: {}/{} done, ETA: {02}:{02}\n", .{ n_done, n_todo, eta_min, eta_s });
            }
        }
    }
}

fn getToken(inp: []const llama.llama_token, draft: []const llama.llama_token, i: usize) llama.llama_token {
    return if (i < inp.len)
        inp[i]
    else
        draft[1 + i - inp.len];
}

pub fn tryDraftStatic(
    nc_static: *const CommonNGramCache,
    ngram_static: CommonNGram,
) llama.llama_token {
    const maybe_part = nc_static.get(ngram_static);
    if (maybe_part == null) {
        return llama.LLAMA_TOKEN_NULL;
    }

    const part_static: CommonNGramCachePart = maybe_part.?;
    var max_count_static: i32 = 0;
    var sum_count_static: i32 = 0;
    var max_token: llama.llama_token = llama.LLAMA_TOKEN_NULL;

    var iter = part_static.iterator();
    while (iter.next()) |entry| {
        const token = entry.key_ptr.*;
        const count = entry.value_ptr.*;

        if (count > max_count_static) {
            max_token = token;
            max_count_static = count;
        }
        sum_count_static += count;
    }

    const threshold_index = LLAMA_NGRAM_STATIC - 1;
    if (sum_count_static < draft_min_sample_size_lax[threshold_index]) {
        return llama.LLAMA_TOKEN_NULL;
    }

    if (100 * max_count_static < draft_min_percent_lax[threshold_index] * sum_count_static) {
        return llama.LLAMA_TOKEN_NULL;
    }

    return max_token;
}

pub fn tryDraftDynamic(
    nc_primary: CommonNGramCache,
    ngrams_primary: []const CommonNGram,
    part_static: CommonNGramCachePart,
    min_sample_size: []const i32,
    min_percent: []const i32,
) llama.llama_token {
    var drafted_token: llama.llama_token = llama.LLAMA_TOKEN_NULL;

    var i: isize = @as(isize, @intCast(ngrams_primary.len)) - 1;
    while (i >= 0 and drafted_token == llama.LLAMA_TOKEN_NULL) : (i -= 1) {
        const ngram_primary = ngrams_primary[i];

        const part_primary_opt = nc_primary.get(ngram_primary);
        if (part_primary_opt == null) {
            i -= 1;
            continue;
        }
        const part_primary = part_primary_opt.?;

        var max_count_primary: i32 = 0;
        var max_count_static: i32 = 0;
        var sum_count_primary: i32 = 0;
        var max_token: llama.llama_token = llama.LLAMA_TOKEN_NULL;

        var iter = part_primary.iterator();
        while (iter.next()) |entry| {
            const token = entry.key_ptr.*;
            const count_primary = entry.value_ptr.*;

            const count_static_opt = part_static.get(token);
            const count_static = if (count_static_opt) |val| 100 * val.* else 1;

            if (count_primary * count_static > max_count_primary * max_count_static) {
                max_token = token;
                max_count_primary = count_primary;
                max_count_static = count_static;
            }

            sum_count_primary += count_primary;
        }

        if (sum_count_primary < min_sample_size[i]) {
            i -= 1;
            continue;
        }
        if (100 * max_count_primary < min_percent[i] * sum_count_primary) {
            i -= 1;
            continue;
        }

        drafted_token = max_token;
    }

    return drafted_token;
}

pub fn common_ngram_cache_draft(
    inp: []llama.llama_token,
    draft: []llama.llama_token,
    n_draft: usize,
    ngram_min: usize,
    ngram_max: usize,
    nc_context: *CommonNGramCache,
    nc_dynamic: *CommonNGramCache,
    nc_static: *CommonNGramCache,
) void {
    // Assert draft size == 1
    std.debug.assert(draft.len == 1);

    const inp_size = inp.len;

    if (inp_size < LLAMA_NGRAM_STATIC) return;

    var draft_tokens = try std.ArrayList(llama.llama_token).initCapacity(draft.len);

    while (draft_tokens.len - 1 < n_draft) {
        var drafted_token: llama.llama_token = llama.LLAMA_TOKEN_NULL;

        const ngram_start_static: usize = inp_size - @as(usize, @intCast(LLAMA_NGRAM_STATIC)) + draft_tokens.items.len - 1;

        // Build static ngram
        var ngram_static = CommonNGram.initEmpty();
        var ngram_static_idx: usize = 0;
        while (ngram_static_idx < LLAMA_NGRAM_STATIC) : (ngram_static_idx += 1) {
            const idx = ngram_start_static + ngram_static_idx;
            ngram_static.tokens[ngram_static_idx] = getToken(inp, draft_tokens.items, idx);
        }

        // Look up static part in static cache
        var part_static = CommonNGramCachePart{};
        if (nc_static.*.get(ngram_static)) |found_part| {
            part_static = found_part.*;
        }

        // Build context + dynamic ngrams vector
        var ngrams_cd: std.ArrayList(CommonNGram) = .empty;
        defer ngrams_cd.deinit();

        var ngram_size_cd: usize = ngram_min;
        while (ngram_size_cd <= ngram_max) : (ngram_size_cd += 1) {
            const ngram_start_cd = inp_size - ngram_size_cd + draft_tokens.items.len - 1;

            var ngram_cd = CommonNGram.initEmpty();
            var j: usize = 0;
            while (j < ngram_size_cd) : (j += 1) {
                const idx = ngram_start_cd + j;
                ngram_cd.tokens[j] = getToken(inp, draft_tokens.items, idx);
            }

            try ngrams_cd.append(ngram_cd);
        }

        if (drafted_token == llama.LLAMA_TOKEN_NULL) {
            drafted_token = tryDraftDynamic(nc_context, ngrams_cd.toSlice(), part_static, draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == llama.LLAMA_TOKEN_NULL) {
            drafted_token = tryDraftDynamic(nc_dynamic, ngrams_cd.toSlice(), part_static, draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == llama.LLAMA_TOKEN_NULL) {
            drafted_token = tryDraftStatic(nc_static, ngram_static);
        }

        if (drafted_token == llama.LLAMA_TOKEN_NULL) {
            break;
        }

        std.debug.print(" - draft candidate: token={}\n", .{drafted_token});
        try draft_tokens.append(drafted_token);
    }
}

pub fn common_ngram_cache_save(
    ngram_cache: *CommonNGramCache,
    filename: []const u8,
) void {
    var file = try std.fs.cwd().createFile(filename, .{ .read = false, .truncate = true });
    defer file.close();
    var writer = file.writer();
    const ngram_cache_it = ngram_cache.iterator();

    while (ngram_cache_it.next()) |item| {
        const ngram = item.key_ptr;
        const token_counts = item.value_ptr;
        std.debug.assert(token_counts.count() > 0);

        const n_tokens = token_counts.count();
        try writer.writeAll(std.mem.asBytes(&ngram));
        try writer.writeAll(std.mem.asBytes(&n_tokens));
        var token_iter = token_counts.iterator();
        while (token_iter.next()) |tok_entry| {
            const token = tok_entry.key_ptr;
            const count = tok_entry.value_ptr;
            std.debug.assert(count > 0);

            try writer.writeAll(std.mem.asBytes(&token));
            try writer.writeAll(std.mem.asBytes(&count));
        }
    }
}

// pub fn common_ngram_cache_load(
//     allocator: *std.mem.Allocator,
//     filename: []const u8,
// ) CommonNGramCache {
//     // Implementation goes here
//     // Use allocator to construct and return a loaded map
//     return undefined;
// }

pub fn common_ngram_cache_merge(
    target: *CommonNGramCache,
    source: *CommonNGramCache,
) void {
    const source_it = source.iterator();
    while (source_it.next()) |entry| {
        const ngram = entry.key_ptr.*;
        const part = entry.value_ptr.*;

        // Check if the ngram exists in target
        const found = try target.getOrPut(ngram);

        if (found.found_existing) {
            if (!found.found_existing) {
                // If not found, copy the whole part directly
                var new_part = try CommonNGramCachePart.init(found.value_ptr.allocator);
                var part_it = part.iterator();
                while (part_it.next()) |tok_count| {
                    try new_part.put(tok_count.key_ptr.*, tok_count.value_ptr.*);
                }
                found.value_ptr.* = new_part;
            } else {
                // Merge token counts
                var part_it = part.iterator();
                while (part_it.next()) |tok_count| {
                    const token = tok_count.key_ptr.*;
                    const count = tok_count.value_ptr.*;
                    std.debug.assert(count > 0);

                    var inner_part = &found.value_ptr.*;
                    const inner_part_found = try inner_part.getOrPut(token);
                    if (inner_part_found) {
                        if (!inner_part_found.found_existing) {
                            inner_part_found.value_ptr.* = count;
                        } else {
                            inner_part_found.value_ptr.* += count;
                        }
                    }
                }
            }
        }
    }
}
