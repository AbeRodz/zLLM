const std = @import("std");
const enums = @import("sampler/enums.zig");
const types = @import("sampler/types.zig");
const RingBuffer = @import("../utils/ring_buffer.zig").RingBuffer;
pub const llama = @cImport({
    @cInclude("llama.h");
});

pub const CommonParamsSampling = struct {
    seed: u32 = llama.LLAMA_DEFAULT_SEED,
    n_prev: i32 = 64,
    n_probs: i32 = 0,
    min_keep: i32 = 0,
    top_k: i32 = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    xtc_probability: f32 = 0.00,
    xtc_threshold: f32 = 0.10,
    typ_p: f32 = 1.00,
    temp: f32 = 0.80,
    dynatemp_range: f32 = 0.00,
    dynatemp_exponent: f32 = 1.00,
    penalty_last_n: i32 = 64,
    penalty_repeat: f32 = 1.00,
    penalty_freq: f32 = 0.00,
    penalty_present: f32 = 0.00,
    dry_multiplier: f32 = 0.0,
    dry_base: f32 = 1.75,
    dry_allowed_length: i32 = 2,
    dry_penalty_last_n: i32 = -1,
    mirostat: i32 = 0,
    top_n_sigma: f32 = -1.00,
    mirostat_tau: f32 = 5.00,
    mirostat_eta: f32 = 0.10,
    ignore_eos: bool = false,
    no_perf: bool = false,
    timing_per_token: bool = false,

    dry_sequence_breakers: []const []const u8 = &.{ "\n", ":", "\"", "*" },

    samplers: []const enums.common_sampler_type = &.{
        .COMMON_SAMPLER_TYPE_NONE,
        .COMMON_SAMPLER_TYPE_DRY,
        .COMMON_SAMPLER_TYPE_TOP_K,
        .COMMON_SAMPLER_TYPE_TOP_P,
        .COMMON_SAMPLER_TYPE_MIN_P,
        .COMMON_SAMPLER_TYPE_TYPICAL_P,
        .COMMON_SAMPLER_TYPE_TEMPERATURE,
        .COMMON_SAMPLER_TYPE_XTC,
        .COMMON_SAMPLER_TYPE_INFILL,
        .COMMON_SAMPLER_TYPE_PENALTIES,
    },

    grammar: []const u8 = "",
    grammar_lazy: bool = false,
    grammar_triggers: []const types.common_grammar_trigger = &.{},
    preserved_tokens: std.AutoHashMap(llama.llama_token, void) = undefined, // or use ArrayList if order matters

    logit_bias: std.ArrayList(llama.llama_logit_bias) = .empty,

    pub fn init(allocator: std.mem.Allocator) !CommonParamsSampling {
        return CommonParamsSampling{
            .preserved_tokens = std.AutoHashMap(llama.llama_token, void).init(allocator),
            // all other fields are default initialized
        };
    }

    pub fn print(self: CommonParamsSampling) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        try out.writer().print(
            "seed: {}, n_prev: {}, n_probs: {}, top_k: {}, top_p: {}\n",
            .{ self.seed, self.n_prev, self.n_probs, self.top_k, self.top_p },
        );
        return out.toOwnedSlice(std.heap.page_allocator) catch "Failed to print params";
    }
};

pub const CommonSampler = struct {
    params: CommonParamsSampling,

    grmr: ?*llama.llama_sampler,
    chain: ?*llama.llama_sampler,

    prev: RingBuffer(llama.llama_token, 64), // You must define or import RingBuffer

    cur: std.ArrayListUnmanaged(llama.llama_token_data),
    cur_p: llama.llama_token_data_array,

    pub fn init(allocator: std.mem.Allocator, model: *const llama.llama_model, params: CommonParamsSampling) !*CommonSampler {
        const vocab = llama.llama_model_get_vocab(model);
        var lparams = llama.llama_sampler_chain_default_params();
        lparams.no_perf = params.no_perf;

        var grmr: ?*llama.llama_sampler = null;

        // Handle grammar initialization
        // if (std.mem.startsWith(u8, params.grammar, "%llguidance")) {
        //     // Ensure LLAMA_USE_LLGUIDANCE is enabled in your build
        //     //grmr = llama.llama_sampler_init_llg(vocab, "lark", params.grammar.ptr);
        // } else {
        var patterns_at_start: std.ArrayList([]const u8) = .empty;
        var patterns_anywhere: std.ArrayList([]const u8) = .empty;
        var trigger_tokens: std.ArrayList(llama.llama_token) = .empty;

        for (params.grammar_triggers) |trigger| {
            switch (trigger.common_grammar_trigger_type) {
                .COMMON_GRAMMAR_TRIGGER_TYPE_WORD => {
                    try patterns_anywhere.append(allocator, try regexEscape(trigger.value));
                },
                .COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, .COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START => {
                    if (trigger.common_grammar_trigger_type == .COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START) {
                        try patterns_at_start.append(allocator, trigger.value);
                    } else {
                        try patterns_anywhere.append(allocator, trigger.value);
                    }
                },
                .COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN => {
                    try trigger_tokens.append(allocator, trigger.token);
                },
            }

            var trigger_patterns: std.ArrayList([]const u8) = .empty;
            if (patterns_at_start.items.len > 0) {
                const pattern = try joinPatterns(allocator, patterns_at_start.items, true);
                try trigger_patterns.append(allocator, pattern);
            }
            if (patterns_anywhere.items.len > 0) {
                const pattern = try joinPatterns(allocator, patterns_anywhere.items, false);
                try trigger_patterns.append(allocator, pattern);
            }

            var trigger_patterns_c: std.ArrayList([*c]const u8) = .empty;
            for (trigger_patterns.items) |regex| {
                try trigger_patterns_c.append(allocator, regex.ptr);
            }

            if (params.grammar_lazy) {
                grmr = llama.llama_sampler_init_grammar_lazy_patterns(
                    vocab,
                    params.grammar.ptr,
                    "root",
                    trigger_patterns_c.items.ptr,
                    @as(u32, @intCast(trigger_patterns_c.items.len)),
                    trigger_tokens.items.ptr,
                    @as(u32, @intCast(trigger_tokens.items.len)),
                );
            } else {
                grmr = llama.llama_sampler_init_grammar(vocab, params.grammar.ptr, "root");
            }

            // if (grmr == null) {
            //     return error.InitializationFailed;
            // }
        }

        const chain = llama.llama_sampler_chain_init(lparams);
        const prev = RingBuffer(llama.llama_token, 64).init(); // should be @max(32, params.n_prev) but comptime
        const cur: std.ArrayListUnmanaged(llama.llama_token_data) = .empty;
        const cur_p = llama.llama_token_data_array{
            .data = null,
            .size = 0,
            .sorted = false,
            .selected = -1,
        };

        const sampler = try allocator.create(CommonSampler);
        sampler.* = CommonSampler{
            .params = params,
            .grmr = grmr,
            .chain = chain,
            .prev = prev,
            .cur = cur,
            .cur_p = cur_p,
        };

        // Add logit bias sampler
        llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_logit_bias(
            llama.llama_vocab_n_tokens(vocab),
            @as(i32, @intCast(params.logit_bias.items.len)),
            params.logit_bias.items.ptr,
        ));

        // Add other samplers based on params
        if (params.mirostat == 0) {
            if (params.top_n_sigma >= 0) {
                llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_top_k(params.top_k));
                llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_temp(params.temp));
                llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_top_n_sigma(params.top_n_sigma));
            } else {
                for (params.samplers) |cnstr| {
                    switch (cnstr) {
                        .COMMON_SAMPLER_TYPE_NONE => {},
                        .COMMON_SAMPLER_TYPE_DRY => {
                            var c_breakers: std.ArrayList([*c]const u8) = .empty;
                            for (params.dry_sequence_breakers) |str| {
                                try c_breakers.append(allocator, str.ptr);
                            }
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_dry(
                                vocab,
                                llama.llama_model_n_ctx_train(model),
                                params.dry_multiplier,
                                params.dry_base,
                                params.dry_allowed_length,
                                params.dry_penalty_last_n,
                                c_breakers.items.ptr,
                                @as(u32, @intCast(c_breakers.items.len)),
                            ));
                        },
                        .COMMON_SAMPLER_TYPE_TOP_K => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_top_k(params.top_k));
                        },
                        .COMMON_SAMPLER_TYPE_TOP_P => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_top_p(params.top_p, @as(usize, @intCast(params.min_keep))));
                        },
                        .COMMON_SAMPLER_TYPE_MIN_P => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_min_p(params.min_p, @as(usize, @intCast(params.min_keep))));
                        },
                        .COMMON_SAMPLER_TYPE_XTC => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, @as(usize, @intCast(params.min_keep)), params.seed));
                        },
                        .COMMON_SAMPLER_TYPE_TYPICAL_P => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_typical(params.typ_p, @as(usize, @intCast(params.min_keep))));
                        },
                        .COMMON_SAMPLER_TYPE_TEMPERATURE => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                        },
                        .COMMON_SAMPLER_TYPE_INFILL => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_infill(vocab));
                        },
                        .COMMON_SAMPLER_TYPE_PENALTIES => {
                            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                        },
                        // else => {
                        //     std.debug.panic("Unknown sampler type");
                        // },
                    }
                }
            }
            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_dist(params.seed));
        } else if (params.mirostat == 1) {
            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_temp(params.temp));
            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_mirostat(llama.llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
        } else if (params.mirostat == 2) {
            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_temp(params.temp));
            llama.llama_sampler_chain_add(sampler.chain, llama.llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
        }

        return sampler;
    }
};
pub fn common_sampler_accept(gsmpl: *CommonSampler, token: llama.llama_token, accept_grammar: bool) void {
    if (accept_grammar) {
        if (gsmpl.grmr) |grmr_ptr| {
            llama.llama_sampler_accept(grmr_ptr, token);
        }
    }

    llama.llama_sampler_accept(gsmpl.chain, token);

    // Append token to the list
    if (!gsmpl.prev.push(token)) {
        // handle allocation error if needed
        std.debug.print("Failed to push token\n", .{});
    }
}
// Helper function to escape regex patterns
fn regexEscape(input: []const u8) ![]const u8 {
    // Implement regex escaping as needed
    return input;
}

// Helper function to join patterns
fn joinPatterns(allocator: std.mem.Allocator, patterns: [][]const u8, at_start: bool) ![]const u8 {
    var joined: std.ArrayList(u8) = .empty;
    if (at_start) {
        try joined.appendSlice(allocator, "^(");
    } else {
        try joined.appendSlice(allocator, "^[\\s\\S]*?(");
    }
    for (0.., patterns) |i, pattern| {
        if (i > 0) {
            try joined.appendSlice(allocator, "|");
        }
        try joined.appendSlice(allocator, pattern);
    }
    try joined.appendSlice(allocator, ")[\\s\\S]*");
    return joined.toOwnedSlice(allocator);
}

pub fn setLogits(allocator: std.mem.Allocator, self: *CommonSampler, ctx: *llama.llama_context, idx: i32) !void {
    const logits = llama.llama_get_logits_ith(ctx, idx);

    const model = llama.llama_get_model(ctx);
    const vocab = llama.llama_model_get_vocab(model);

    const n_vocab = llama.llama_vocab_n_tokens(vocab);

    try self.cur.resize(allocator, @as(usize, @intCast(n_vocab)));

    var token_id: llama.llama_token = 0;
    while (token_id < n_vocab) : (token_id += 1) {
        self.cur.items[@as(usize, @intCast(token_id))] = llama.llama_token_data{
            .id = token_id,
            .logit = logits[@as(usize, @intCast(token_id))],
            .p = 0.0,
        };
    }

    self.cur_p = llama.llama_token_data_array{
        .data = self.cur.items.ptr,
        .size = self.cur.items.len,
        .selected = -1,
        .sorted = false,
    };
}

pub fn common_sampler_sample(allocator: std.mem.Allocator, self: *CommonSampler, ctx: *llama.llama_context, idx: i32, grammar_first: bool) llama.llama_token {
    setLogits(allocator, self, ctx, idx) catch unreachable;

    const chain = self.chain orelse unreachable;
    const cur_p = &self.cur_p;

    // Apply grammar first (only when a grammar sampler exists).
    if (grammar_first) {
        if (self.grmr) |grmr| llama.llama_sampler_apply(grmr, cur_p);
    }

    llama.llama_sampler_apply(chain, cur_p);
    std.debug.assert(cur_p.selected != -1);

    const id = cur_p.data[@as(usize, @intCast(cur_p.selected))].id;

    // If grammar was applied first, we're done.
    if (grammar_first) return id;

    // No grammar → just return the sampled token.
    const grmr = self.grmr orelse return id;

    // Grammar validation: check whether the sampled token is grammar-valid.
    var single_token_data = llama.llama_token_data{ .id = id, .logit = 1.0, .p = 0.0 };
    var single_token_data_array = llama.llama_token_data_array{
        .data = &single_token_data,
        .size = 1,
        .sorted = false,
        .selected = -1,
    };
    llama.llama_sampler_apply(grmr, &single_token_data_array);

    const is_valid = single_token_data_array.data[0].logit != -std.math.inf(f32);
    if (is_valid) return id;

    // Token failed grammar check — resample with grammar applied first.
    setLogits(allocator, self, ctx, idx) catch unreachable;
    llama.llama_sampler_apply(grmr, &self.cur_p);
    llama.llama_sampler_apply(chain, &self.cur_p);
    std.debug.assert(self.cur_p.selected != -1);
    return self.cur_p.data[@as(usize, @intCast(self.cur_p.selected))].id;
}
