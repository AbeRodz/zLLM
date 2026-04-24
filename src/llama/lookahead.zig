const std = @import("std");
const types = @import("sampler/types.zig");
const common_types = @import("./common/types.zig");
const enums = @import("sampler/enums.zig");
const common_init = @import("llama_common_init.zig");
const common_sampler = @import("llama_sampler.zig");
const common = @import("llama_common.zig");
const llama = @cImport({
    @cInclude("llama.h");
});

pub const NgramData = struct {
    actve: bool = false,
    seq_id: llama.llama_seq_id = -1,
    i_batch: std.ArrayListUnmanaged(i32),
    tokens: std.ArrayListUnmanaged(llama.llama_token),
};

pub const NgramContainer = struct {
    allocator: std.mem.Allocator,
    n_total: i32 = 0,
    count: std.ArrayListUnmanaged(i32),
    head: std.ArrayListUnmanaged(i32),
    tokens: std.ArrayListUnmanaged(llama.llama_token),

    // pub fn init(
    //     allocator: std.mem.Allocator,
    //     n_vocab: i32,
    //     N: i32,
    //     G: i32,
    // ) !NgramContainer {
    //     return NgramContainer{
    //         .count = std.ArrayList(i32).initCapacity(
    //             allocator,
    //             n_vocab,
    //         ),
    //         .head = std.ArrayList(i32).init(
    //             allocator,
    //             n_vocab,
    //         ),
    //         .tokens = std.ArrayList(llama.llama_token).initCapacity(
    //             allocator,
    //             n_vocab * G * (N - 1),
    //         ),
    //     };
    // }
    pub fn init(
        allocator: std.mem.Allocator,
        n_vocab: usize,
        N: usize,
        G: usize,
    ) !NgramContainer {
        var count = try std.ArrayListUnmanaged(i32).initCapacity(allocator, n_vocab);
        var head = try std.ArrayListUnmanaged(i32).initCapacity(allocator, n_vocab);
        var tokens = try std.ArrayListUnmanaged(llama.llama_token)
            .initCapacity(allocator, n_vocab * G * (N - 1));

        // MATCH C++ resize()
        try count.resize(allocator, n_vocab);
        try head.resize(allocator, n_vocab);
        try tokens.resize(allocator, n_vocab * G * (N - 1));

        // zero-init like std::vector
        @memset(count.items, 0);
        @memset(head.items, 0);
        @memset(tokens.items, 0);

        return .{
            .allocator = allocator,
            .n_total = 0,
            .count = count,
            .head = head,
            .tokens = tokens,
        };
    }
    pub fn deinit(self: *NgramContainer) void {
        self.count.deinit(self.allocator);
        self.head.deinit(self.allocator);
        self.tokens.deinit(self.allocator);
    }
};
pub fn look(model_path: []const u8, prompt: []const u8, allocator: std.mem.Allocator) !void {
    const W: usize = 3;
    const N: usize = 8;
    const G: usize = 3;

    var params: common_types.CommonParams = .{};
    params.model.path = model_path;
    params.prompt = prompt;
    // Lookahead needs W + G + 1 simultaneous sequences in the KV cache.
    params.n_parallel = @intCast(W + G + 1);
    // Disable automatic KV defragmentation: lookahead manages sequences
    // explicitly (seq_rm / seq_keep / seq_cp) every decode step, so letting
    // llama.cpp defrag concurrently causes graph reallocation conflicts.
    params.defrag_thold = -1.0;
    // DRY penalty scans the last n tokens on every sample call.
    // With W*(N-1) positions sampled per step this becomes O(n_ctx) per step.
    params.sampling.dry_penalty_last_n = 0;
    // Skip the dummy warmup decode — saves ~1 s on cold start.
    params.warmup = false;

    // --- KV cache sizing ---
    // Each lookahead step holds (W+G+1) simultaneous sequences, each needing
    // up to n_past + N positions.  Total KV slots ≈ 31 * (n_predict + N).
    // n_ctx=4096 overflows at ~132 tokens; 16 384 supports ~520 tokens.
    params.n_ctx = @intCast((W + G + 2) * 512); // 16 384

    // Max tokens in a single decode call:
    //   1 (current) + G*(N-1) (verify) + (W-1) + (N-2)*W (lookahead) ≈ 120
    // Keep n_ubatch ≥ that so llama.cpp never splits the batch mid-step.
    params.n_ubatch = @intCast(1 + G * (N - 1) + (W - 1) + (N - 2) * W + 32);
    params.n_batch = params.n_ubatch;

    // Quantised KV cache: Q8_0 halves VRAM vs F16 with negligible quality loss.
    // params.cache_type_k = common_types.llama.GGML_TYPE_Q8_0;
    // params.cache_type_v = common_types.llama.GGML_TYPE_Q8_0;

    // Flash attention: fused QK^T V kernel — big win on Metal / CUDA.
    params.flash_attn = false;

    llama.llama_backend_init();
    llama.llama_numa_init(params.numa);

    const init = try common_init.common_init_from_params(allocator, &params);
    const model = init.model;
    const ctx = init.context;
    const vocab = llama.llama_model_get_vocab(@ptrCast(model)).?;

    // -------------------------------
    // Tokenize prompt
    // -------------------------------
    var inp = try common.common_tokenize(
        allocator,
        @ptrCast(vocab),
        params.prompt,
        true,
        true,
    );
    defer inp.deinit(allocator);

    const max_ctx = llama.llama_n_ctx(@ptrCast(ctx));
    if (inp.items.len > max_ctx - 4) {
        return error.InputTooLong;
    }

    for (inp.items) |t| {
        const piece = try common_init.common_token_to_piece_vocab(allocator, @ptrCast(vocab), t, false);
        std.debug.print("{s}", .{piece});
    }
    std.debug.print("\n", .{});

    // -------------------------------
    // Evaluate prompt
    // -------------------------------
    _ = llama.llama_decode(
        @ptrCast(ctx),
        llama.llama_batch_get_one(inp.items.ptr, @intCast(inp.items.len - 1)),
    );
    _ = llama.llama_decode(
        @ptrCast(ctx),
        llama.llama_batch_get_one(&inp.items[inp.items.len - 1], 1),
    );

    for (1..W + G + 1) |s| {
        llama.llama_kv_self_seq_cp(@ptrCast(ctx), 0, @intCast(s), -1, -1);
    }

    // -------------------------------
    // Init sampler TODO
    // -------------------------------
    const sampler = try common_sampler.CommonSampler.init(allocator, @ptrCast(model.?), params.sampling);
    //defer common.common_sampler_free(sampler);

    // -------------------------------
    // Init batch
    // -------------------------------
    var batch = llama.llama_batch_init(
        params.n_ubatch, // capacity = max tokens per single decode step
        0,
        @intCast(W + G + 1),
    );
    defer llama.llama_batch_free(batch);

    // -------------------------------
    // Init ngram containers
    // -------------------------------
    var ngrams_observed = try NgramContainer.init(
        allocator,
        @intCast(llama.llama_vocab_n_tokens(vocab)),
        N,
        G,
    );
    defer ngrams_observed.deinit();

    var ngrams_cur = try allocator.alloc(NgramData, G);
    defer {
        for (ngrams_cur) |*ng| {
            ng.i_batch.deinit(allocator);
            ng.tokens.deinit(allocator);
        }
        allocator.free(ngrams_cur);
    }

    for (ngrams_cur) |*ng| {
        ng.* = .{
            .actve = false,
            .seq_id = -1,
            .i_batch = .empty,
            .tokens = .empty,
        };
    }

    // -------------------------------
    // Init lookahead tokens
    // -------------------------------
    const tokens_j_prev = try allocator.alloc(llama.llama_token, W);
    defer allocator.free(tokens_j_prev);

    const tokens_j = try allocator.alloc([]llama.llama_token, N - 1);
    defer {
        for (tokens_j) |row| allocator.free(row);
        allocator.free(tokens_j);
    }

    for (tokens_j) |*row| {
        row.* = try allocator.alloc(llama.llama_token, W);
        for (row.*, 0..) |*t, i| t.* = @intCast(100 + i);
    }

    const seq_id_all = try allocator.alloc(llama.llama_seq_id, W + G + 1);
    defer allocator.free(seq_id_all);
    for (seq_id_all, 0..) |*s, i| s.* = @intCast(i);

    // -------------------------------
    // Sample first token
    // -------------------------------
    var id = common_sampler.common_sampler_sample(allocator, sampler, @ptrCast(ctx.?), 0, false);
    common_sampler.common_sampler_accept(sampler, id, true);

    std.debug.print(
        "{s}",
        .{try common_init.common_token_to_piece_vocab(allocator, @ptrCast(vocab), id, false)},
    );

    var n_past: i32 = @intCast(inp.items.len);
    var n_predict: i32 = 0;
    var n_accept: i32 = 0;
    var has_eos = false;

    const t_dec_start = std.time.nanoTimestamp();

    // ==========================================================
    // MAIN LOOP
    // ==========================================================
    while (true) {
        common.common_batch_clear(@ptrCast(&batch));

        // Current token
        common.common_batch_add(
            @ptrCast(&batch),
            id,
            n_past,
            seq_id_all,
            true,
        );

        // -------------------------------
        // Verification n-grams
        // -------------------------------
        const g_cur = ngrams_observed.count.items[@intCast(id)];
        for (ngrams_cur[0..@intCast(g_cur)], 0..) |*ng, g| {
            ng.actve = true;
            ng.seq_id = @intCast(W + 1 + g);
            ng.tokens.clearRetainingCapacity();
            ng.i_batch.clearRetainingCapacity();

            try ng.tokens.append(allocator, id);
            try ng.i_batch.append(allocator, 0);

            for (0..N - 1) |j| {
                const idx =
                    @as(usize, @intCast(id)) * (N - 1) * G + g * (N - 1) + j;
                const t = ngrams_observed.tokens.items[idx];
                try ng.tokens.append(allocator, t);
                try ng.i_batch.append(allocator, @intCast(batch.n_tokens));

                common.common_batch_add(
                    @ptrCast(&batch),
                    t,
                    n_past + @as(i32, @intCast(j + 1)),
                    &[_]llama.llama_seq_id{ng.seq_id},
                    true,
                );
            }
        }

        // -------------------------------
        // Lookahead levels
        // -------------------------------
        for (1..W) |i| {
            const seqs = try allocator.alloc(llama.llama_seq_id, W - i);
            defer allocator.free(seqs);

            for (seqs, 0..) |*s, j| s.* = @intCast(i + j + 1);

            common.common_batch_add(
                @ptrCast(&batch),
                tokens_j[0][i],
                n_past + @as(i32, @intCast(i)),
                seqs,
                false,
            );
        }

        for (1..N - 1) |j| {
            for (0..W) |i| {
                common.common_batch_add(
                    @ptrCast(&batch),
                    tokens_j[j][i],
                    n_past + @as(i32, @intCast(j + i)),
                    &[_]llama.llama_seq_id{@intCast(i + 1)},
                    j == N - 2,
                );
            }
        }

        if (llama.llama_decode(@ptrCast(ctx.?), batch) != 0) {
            return error.DecodeFailed;
        }

        var seq_id_best: llama.llama_seq_id = 0;

        // -------------------------------
        // Acceptance loop
        // -------------------------------
        for (0..N) |v| {
            var i_batch: i32 = 0;

            if (v > 0) {
                for (ngrams_cur[0..@intCast(g_cur)]) |ng| {
                    if (ng.actve) {
                        i_batch = ng.i_batch.items[v];
                        seq_id_best = ng.seq_id;
                        n_accept += 1;
                        break;
                    }
                }
                if (i_batch == 0) break;
            }

            id = common_sampler.common_sampler_sample(allocator, sampler, @ptrCast(ctx.?), i_batch, false);
            common_sampler.common_sampler_accept(sampler, id, true);

            // Verified tokens print in cyan; the main token prints normally.
            const piece = try common_init.common_token_to_piece_vocab(allocator, @ptrCast(vocab), id, false);
            if (v == 0) {
                std.debug.print("{s}", .{piece});
            } else {
                std.debug.print("\x1b[0;96m{s}\x1b[0m", .{piece});
            }

            n_predict += 1;
            n_past += 1;

            // Fixed: was incorrectly inverted with `!`
            if (llama.llama_vocab_is_eog(@ptrCast(vocab), id)) {
                has_eos = true;
                break;
            }

            // Deactivate n-grams that no longer match.
            for (ngrams_cur[0..@intCast(g_cur)]) |*ng| {
                if (ng.actve) {
                    if (v == N - 1 or ng.tokens.items[v + 1] != id) {
                        ng.actve = false;
                    }
                }
            }

            // -------------------------------
            // Update lookahead tokens
            // Save the outgoing first level, shift levels up, fill new last level.
            // -------------------------------
            for (0..W) |i| tokens_j_prev[i] = tokens_j[0][i];
            // Shift: tokens_j[0] = tokens_j[1], ..., tokens_j[N-3] = tokens_j[N-2]
            for (0..N - 2) |j| @memcpy(tokens_j[j], tokens_j[j + 1]);

            if (v == 0) {
                // Sample new last-level tokens from the batch logits.
                // Batch layout after position 0:
                //   g_cur*(N-1) verification tokens, then (W-1) + (N-2)*W lookahead tokens.
                // Last level (j = N-2 in the 1..N-1 batch loop) starts at index:
                //   g_cur*(N-1) + (W-1) + (N-3)*W  =  g_cur*(N-1) + W*(N-2)
                const g_cur_sz: usize = @intCast(g_cur);
                for (0..W) |i| {
                    tokens_j[N - 2][i] = common_sampler.common_sampler_sample(
                        allocator,
                        sampler,
                        @ptrCast(ctx.?),
                        @intCast(g_cur_sz * (N - 1) + W * (N - 2) + i),
                        false,
                    );
                }
            } else {
                // Verified path: re-initialize last level from the (now shifted) first level.
                for (0..W) |i| tokens_j[N - 2][i] = tokens_j[0][i];
            }

            // -------------------------------
            // Update observed n-grams
            // Only done on the main decode step (v == 0) to avoid duplicates.
            // Ref: https://github.com/hao-ai-lab/LookaheadDecoding/issues/14#issuecomment-1826198518
            // -------------------------------
            if (v == 0) {
                for (0..W) |f| {
                    const ft: usize = @intCast(tokens_j_prev[f]); // key: first token of n-gram

                    // Build the n-gram from the shifted tokens_j.
                    var ngram: [N - 1]llama.llama_token = undefined;
                    for (0..N - 1) |j| ngram[j] = tokens_j[j][f];

                    // Skip if this exact n-gram is already stored.
                    const cnt: usize = @intCast(ngrams_observed.count.items[ft]);
                    var is_unique = true;
                    for (0..cnt) |k| {
                        const base = ft * (N - 1) * G + k * (N - 1);
                        var match = true;
                        for (0..N - 1) |j| {
                            if (ngrams_observed.tokens.items[base + j] != ngram[j]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            is_unique = false;
                            break;
                        }
                    }
                    if (!is_unique) continue;

                    // Write into the ring buffer at head position.
                    const head: usize = @intCast(ngrams_observed.head.items[ft]);
                    const base = ft * (N - 1) * G + head * (N - 1);
                    for (0..N - 1) |j| ngrams_observed.tokens.items[base + j] = ngram[j];

                    ngrams_observed.count.items[ft] = @intCast(@min(G, cnt + 1));
                    ngrams_observed.head.items[ft] = @intCast((head + 1) % G);
                    ngrams_observed.n_total += 1;
                }
            }
        }

        if (has_eos) break;

        // -------------------------------
        // KV cache management
        // -------------------------------
        _ = llama.llama_kv_self_seq_rm(@ptrCast(ctx.?), -1, n_past, -1);

        if (seq_id_best != 0) {
            llama.llama_kv_self_seq_keep(@ptrCast(ctx.?), seq_id_best);
            llama.llama_kv_self_seq_cp(@ptrCast(ctx.?), seq_id_best, 0, -1, -1);
            _ = llama.llama_kv_self_seq_rm(@ptrCast(ctx.?), seq_id_best, -1, -1);

            for (1..W + G + 1) |s| {
                llama.llama_kv_self_seq_cp(@ptrCast(ctx.?), 0, @intCast(s), -1, -1);
            }
        }
    }

    const t_dec_end = std.time.nanoTimestamp();
    const dec_s = @as(f64, @floatFromInt(t_dec_end - t_dec_start)) / 1e9;

    std.debug.print("\n\n", .{});
    std.debug.print("decoded {d} tokens in {d:.3} s, speed: {d:.3} t/s\n", .{
        n_predict,
        dec_s,
        @as(f64, @floatFromInt(n_predict)) / dec_s,
    });
    std.debug.print("W = {d}  N = {d}  G = {d}\n", .{ W, N, G });
    std.debug.print("n_predict = {d}\n", .{n_predict});
    std.debug.print("n_accept  = {d}  ({d:.1}% acceptance rate)\n", .{
        n_accept,
        if (n_predict > 0) @as(f64, @floatFromInt(n_accept)) / @as(f64, @floatFromInt(n_predict)) * 100.0 else 0.0,
    });
}

test "lookahead" {
    // Requires a real model path — run manually with `zig build run -- run-lookahead <model> "<prompt>"`
    try look("/tmp/model.gguf", "Once upon a time", std.testing.allocator);
}
