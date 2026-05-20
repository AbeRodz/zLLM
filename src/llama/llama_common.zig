const std = @import("std");
const types = @import("./common/types.zig");
pub const llama = @cImport({
    @cInclude("llama.h");
});

// Use the local cImport type to avoid cross-module @cImport type conflicts.
pub const model_params = llama.llama_model_params;

pub fn common_batch_add(
    batch: *llama.llama_batch,
    id: llama.llama_token,
    pos: llama.llama_pos,
    seq_ids: []const llama.llama_seq_id,
    logits: bool,
) void {
    const b_ntokens = @as(usize, @intCast(batch.n_tokens));

    batch.token[b_ntokens] = id;
    batch.pos[b_ntokens] = pos;
    batch.n_seq_id[b_ntokens] = @as(i32, @intCast(seq_ids.len));

    // Copy seq_ids into batch.seq_id[i]
    // assuming batch.seq_id is [][]llama_seq_id, fixed-size inner arrays
    // and seq_ids.len <= capacity of batch.seq_id[i]
    for (0.., seq_ids) |i, _| {
        batch.seq_id[b_ntokens][i] = seq_ids[i];
    }

    batch.logits[b_ntokens] = @intFromBool(logits);

    batch.n_tokens += 1;
}

pub fn common_batch_clear(batch: *llama.llama_batch) void {
    batch.n_tokens = 0;
}

/// Returns llama model params as a value — do NOT return a pointer to a local.
pub fn commonModelParamsToLlama(params: *const types.CommonParams) model_params {
    var mparams = llama.llama_model_default_params();

    if (params.devices.len != 0) {
        mparams.devices = params.devices.ptr;
    }

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }

    mparams.main_gpu = params.main_gpu;
    mparams.split_mode = params.split_mode;
    mparams.tensor_split = params.tensor_split[0..].ptr;
    mparams.use_mmap = params.use_mmap;
    mparams.use_mlock = params.use_mlock;
    mparams.check_tensors = params.check_tensors;

    if (params.kv_overrides.items.len == 0) {
        mparams.kv_overrides = null;
    } else {
        std.debug.assert(params.kv_overrides.items[params.kv_overrides.items.len - 1].key[0] == 0);
        mparams.kv_overrides = @ptrCast(params.kv_overrides.items.ptr);
    }

    if (params.tensor_buft_overrides.items.len == 0) {
        mparams.tensor_buft_overrides = null;
    } else {
        std.debug.assert(params.tensor_buft_overrides.items[params.tensor_buft_overrides.items.len - 1].pattern == null);
    }

    return mparams;
}

pub fn common_tokenize_context(
    allocator: std.mem.Allocator,
    ctx: llama.llama_context,
    text: []const u8,
    add_special: bool,
    parse_special: bool,
) !std.ArrayList(llama.llama_token) {
    const model = llama.llama_get_model(ctx).?;
    const vocab = llama.llama_model_get_vocab(model).?;
    return common_tokenize(allocator, vocab, text, add_special, parse_special);
}

/// Tokenize `text` into a caller-owned ArrayListUnmanaged.
/// Caller must call result.deinit(allocator) when done.
pub fn common_tokenize(
    allocator: std.mem.Allocator,
    vocab: *const llama.llama_vocab,
    text: []const u8,
    add_special: bool,
    parse_special: bool,
) !std.ArrayListUnmanaged(llama.llama_token) {
    // Initial capacity guess: text byte length + 2 for BOS/EOS.
    const n_initial: usize = text.len + (@as(usize, @intFromBool(add_special)) * 2);
    var result = try std.ArrayListUnmanaged(llama.llama_token).initCapacity(allocator, n_initial);
    errdefer result.deinit(allocator);

    // First attempt using the full allocated capacity as the output buffer.
    var n_tokens = llama.llama_tokenize(
        vocab,
        text.ptr,
        @as(i32, @intCast(text.len)),
        result.items.ptr,
        @as(i32, @intCast(result.capacity)), // capacity, not items.len (which is 0)
        add_special,
        parse_special,
    );

    if (n_tokens < 0) {
        // llama returns -(needed_size) when the buffer is too small.
        const needed: usize = @intCast(-n_tokens);
        try result.resize(allocator, needed);
        n_tokens = llama.llama_tokenize(
            vocab,
            text.ptr,
            @as(i32, @intCast(text.len)),
            result.items.ptr,
            @as(i32, @intCast(result.items.len)),
            add_special,
            parse_special,
        );
        if (n_tokens < 0) return error.TokenizationFailed;
    }

    result.items.len = @intCast(n_tokens);
    return result;
}

pub fn resizeResult(allocator: std.mem.Allocator, result: *std.ArrayList(i32), n_tokens: i32) !void {
    const new_len = @as(usize, @intCast(-n_tokens)); // convert to positive usize

    // resize list
    try result.resize(allocator, new_len);

    // initialize new elements to 0.0
    const start = result.items.len - new_len; // index of first new element
    for (result.items[start..]) |*elem| {
        elem.* = 0.0;
    }
}
