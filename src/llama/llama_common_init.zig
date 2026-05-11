const std = @import("std");
const llama_model = @import("cTypes.zig");
const commonTypes = @import("common/types.zig");
const llama_common = @import("llama_common.zig");
pub const llama = @cImport({
    @cInclude("llama.h");
});

pub const ggml = @cImport({
    @cInclude("ggml.h");
});

pub const gguf = @cImport({
    @cInclude("gguf.h");
});

pub const CommonInitResult = struct {
    model: ?*llama.llama_model,
    context: ?*llama.llama_context,
    lora: std.ArrayList(*llama.llama_adapter_lora),
};

pub fn common_set_adapter_lora(ctx: *llama.llama_context, adapters: std.ArrayList(commonTypes.common_adapter_lora_info)) void {
    for (adapters.items) |adapter| {
        if (adapter.scale != 0.0) {
            _ = llama.llama_set_adapter_lora(ctx, @ptrCast(adapter.ptr), adapter.scale);
        }
    }
}

pub fn common_token_to_piece_vocab(
    allocator: std.mem.Allocator,
    vocab: *const llama.llama_vocab,
    token: llama.llama_token,
    special: bool,
) ![]const u8 {
    var piece = try allocator.alloc(u8, 256);
    errdefer allocator.free(piece);

    var n_chars = llama.llama_token_to_piece(
        vocab,
        token,
        @as([*c]u8, @ptrCast(&piece[0])),
        @as(i32, @intCast(piece.len)),
        0,
        special,
    );

    if (n_chars < 0) {
        const needed: usize = @intCast(-n_chars);
        piece = try allocator.realloc(piece, needed);
        n_chars = llama.llama_token_to_piece(
            vocab,
            token,
            @as([*c]u8, @ptrCast(&piece[0])),
            @as(i32, @intCast(piece.len)),
            0,
            special,
        );
        std.debug.assert(n_chars == @as(i32, @intCast(piece.len)));
        return piece;
    }

    // Shrink the allocation to the actual content length so callers can
    // free() with the correct size. realloc(p, 0) is a valid no-op free in
    // Zig's allocator interface and allocator.free on the returned empty
    // slice is also a no-op, so the zero-char case is handled correctly.
    piece = try allocator.realloc(piece, @intCast(n_chars));
    return piece;
}

pub fn common_token_to_piece_ctx(allocator: std.mem.Allocator, ctx: *llama.llama_context, token: llama.llama_token, special: bool) ![]const u8 {
    const vocab = llama.llama_model_get_vocab(llama.llama_get_model(ctx));
    return common_token_to_piece_vocab(allocator, vocab.?, token, special);
}
pub fn common_init_from_params(allocator: std.mem.Allocator, params: *commonTypes.CommonParams) !CommonInitResult {
    var iparams = CommonInitResult{
        .model = null,
        .context = null,
        .lora = .empty,
    };

    const mparams = llama_common.commonModelParamsToLlama(params);

    // Call the C function directly through this file's own @cImport to avoid
    // cross-module cImport type conflicts.  @bitCast is safe because both
    // instances of llama_model_params are compiled from the same C header and
    // have identical layout.
    const model = llama.llama_model_load_from_file(
        params.model.path.ptr,
        @bitCast(mparams),
    );
    if (model == null) {
        std.log.err("Failed to load model '{s}'", .{params.model.path});
        return iparams;
    }

    const vocab = llama.llama_model_get_vocab(@ptrCast(model));

    if (params.reranking) {
        var ok = true;

        if (llama.llama_vocab_bos(vocab) == llama.LLAMA_TOKEN_NULL) {
            std.log.warn("Vocab does not have a BOS token; reranking will not work", .{});
            ok = false;
        }

        if (llama.llama_vocab_eos(vocab) == llama.LLAMA_TOKEN_NULL) {
            std.log.warn("Vocab does not have an EOS token; reranking will not work", .{});
            ok = false;
        }

        if (llama.llama_vocab_sep(vocab) == llama.LLAMA_TOKEN_NULL) {
            std.log.warn("Vocab does not have a SEP token; reranking will not work", .{});
            ok = false;
        }

        if (!ok) {
            llama.llama_model_free(@ptrCast(model));
            return iparams;
        }
    }

    const cparams = commonContextParamsToLlama(params);

    const ctx = llama.llama_init_from_model(@ptrCast(model), cparams);
    if (ctx == null) {
        std.log.err("Failed to create context with model '{s}'", .{params.model.path});
        llama.llama_model_free(@ptrCast(model));
        return iparams;
    }

    if (params.ctx_shift and !llama.llama_kv_self_can_shift(ctx)) {
        std.log.warn("KV cache shifting is not supported for this context; disabling KV cache shifting", .{});
        params.ctx_shift = false;
    }

    if (params.control_vectors.items.len != 0) {
        if (params.control_vector_layer_start <= 0) {
            params.control_vector_layer_start = 1;
        }
        if (params.control_vector_layer_end <= 0) {
            params.control_vector_layer_end = llama.llama_model_n_layer(@ptrCast(model));
        }

        const cvec = try common_control_vector_load(allocator, params.control_vectors);
        if (cvec.n_embd == -1) {
            llama.llama_free(ctx);
            llama.llama_model_free(@ptrCast(model));
            return iparams;
        }

        const err = llama.llama_apply_adapter_cvec(
            ctx,
            cvec.data.items.ptr,
            cvec.data.items.len,
            cvec.n_embd,
            params.control_vector_layer_start,
            params.control_vector_layer_end,
        );
        if (err != 0) {
            llama.llama_free(ctx);
            llama.llama_model_free(@ptrCast(model));
            return iparams;
        }
    }

    for (params.lora_adapters.items) |*adapter| { // pointer to item
        const lora = llama.llama_adapter_lora_init(@ptrCast(model), adapter.path.ptr);
        if (lora == null) {
            std.log.err("Failed to apply lora adapter '{s}'", .{adapter.path});
            llama.llama_free(ctx);
            llama.llama_model_free(@ptrCast(model));
            return iparams;
        }

        iparams.lora.append(allocator, lora.?) catch {
            adapter.ptr = @ptrCast(lora); // ✅ now mutable
            std.log.err("Failed to append lora adapter", .{});
            llama.llama_free(ctx);
            llama.llama_model_free(@ptrCast(model));
            return iparams;
        };
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(ctx.?, params.lora_adapters);
    }

    if (params.sampling.ignore_eos and llama.llama_vocab_eos(vocab) == llama.LLAMA_TOKEN_NULL) {
        std.log.warn("Vocab does not have an EOS token; ignoring --ignore-eos", .{});
        params.sampling.ignore_eos = false;
    }

    if (params.sampling.ignore_eos) {
        var i: llama.llama_token = 0;
        while (i < llama.llama_vocab_n_tokens(vocab)) : (i += 1) {
            if (llama.llama_vocab_is_eog(vocab, i)) {
                const piece = try common_token_to_piece_ctx(allocator, ctx.?, i, true);
                std.log.info("Added {s} logit bias = -inf", .{piece});
                params.sampling.logit_bias.append(allocator, .{ .token = i, .bias = -std.math.inf(f32) }) catch {
                    std.log.err("Failed to append logit bias", .{});
                    llama.llama_free(ctx);
                    llama.llama_model_free(@ptrCast(model));
                    return iparams;
                };
            }
        }
    }

    if (params.sampling.penalty_last_n == -1) {
        const ctx_size = llama.llama_n_ctx(ctx);
        std.log.info("Setting penalty_last_n to ctx_size = {d}", .{ctx_size});
        params.sampling.penalty_last_n = @as(i32, @intCast(ctx_size));
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        const ctx_size = llama.llama_n_ctx(ctx);
        std.log.info("Setting dry_penalty_last_n to ctx_size = {d}", .{ctx_size});
        params.sampling.dry_penalty_last_n = @as(i32, @intCast(ctx_size));
    }

    if (params.warmup) {
        std.log.warn("Warming up the model with an empty run - please wait ... (--no-warmup to disable)", .{});

        llama.llama_set_warmup(ctx, true);

        var tmp: std.ArrayList(llama.llama_token) = .empty;
        defer tmp.deinit(allocator);

        const bos = llama.llama_vocab_bos(vocab);
        const eos = llama.llama_vocab_eos(vocab);

        if (bos != llama.LLAMA_TOKEN_NULL) {
            tmp.append(allocator, bos) catch {};
        }
        if (eos != llama.LLAMA_TOKEN_NULL) {
            tmp.append(allocator, eos) catch {};
        }
        if (tmp.items.len == 0) {
            tmp.append(allocator, 0) catch {};
        }

        if (llama.llama_model_has_encoder(@ptrCast(model))) {
            const batch = llama.llama_batch_get_one(tmp.items.ptr, @as(i32, @intCast(tmp.items.len)));
            _ = llama.llama_encode(ctx, batch);
            var decoder_start_token_id = llama.llama_model_decoder_start_token(@ptrCast(model));
            if (decoder_start_token_id == llama.LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clearRetainingCapacity();
            tmp.append(allocator, decoder_start_token_id) catch {};
        }
        if (llama.llama_model_has_decoder(@ptrCast(model))) {
            const batch = llama.llama_batch_get_one(tmp.items.ptr, @as(i32, @intCast(@min(tmp.items.len, @as(usize, @intCast(params.n_batch))))));
            _ = llama.llama_decode(ctx, batch);
        }
        llama.llama_kv_self_clear(ctx);
        llama.llama_synchronize(ctx);
        llama.llama_perf_context_reset(ctx);
        llama.llama_set_warmup(ctx, false);
    }

    // Free old pointers if they exist
    if (iparams.model) |old_model| {
        llama.llama_model_free(old_model);
    }
    if (iparams.context) |old_ctx| {
        llama.llama_free(old_ctx);
    }

    // Take ownership of new pointers
    iparams.model = @ptrCast(model); // model must be type ?*llama_model
    iparams.context = ctx; // ctx must be type ?*llama_context

    return iparams;
}

pub fn commonContextParamsToLlama(params: *commonTypes.CommonParams) llama.llama_context_params {
    var cparams = llama.llama_context_default_params();

    cparams.n_ctx = @as(u32, @intCast(params.n_ctx));
    cparams.n_seq_max = @as(u32, @intCast(params.n_parallel));
    cparams.n_batch = @as(u32, @intCast(params.n_batch));
    cparams.n_ubatch = @as(u32, @intCast(params.n_ubatch));
    cparams.n_threads = params.cpuparams.n_threads;
    cparams.n_threads_batch = if (params.cpuparams_batch.n_threads == -1)
        params.cpuparams.n_threads
    else
        params.cpuparams_batch.n_threads;

    cparams.logits_all = params.logits_all;
    cparams.embeddings = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base = params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale;
    cparams.yarn_ext_factor = params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor;
    cparams.yarn_beta_fast = params.yarn_beta_fast;
    cparams.yarn_beta_slow = params.yarn_beta_slow;
    cparams.yarn_orig_ctx = @as(u32, @intCast(params.yarn_orig_ctx));
    cparams.pooling_type = params.pooling_type;
    cparams.attention_type = params.attention_type;
    cparams.defrag_thold = params.defrag_thold;
    cparams.cb_eval = @ptrCast(params.cb_eval);
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv = !params.no_kv_offload;
    cparams.flash_attn = params.flash_attn;
    cparams.no_perf = params.no_perf;

    if (params.reranking) {
        cparams.embeddings = true;
        cparams.pooling_type = llama.LLAMA_POOLING_TYPE_RANK;
    }

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    return cparams;
}

pub fn common_control_vector_load(
    allocator: std.mem.Allocator,
    load_info: std.ArrayList(commonTypes.common_control_vector_load_info),
) !commonTypes.common_control_vector_data {
    var result = commonTypes.common_control_vector_data{
        .n_embd = -1,
        .data = .empty,
    };

    for (load_info.items) |info| {
        const cvec = try common_control_vector_load_one(std.heap.page_allocator, info);
        if (cvec.n_embd == -1) {
            result.n_embd = -1;
            result.data.deinit(allocator);
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = cvec.n_embd;
        } else if (cvec.n_embd != result.n_embd) {
            std.log.err("Control vector dimension mismatch", .{});
            result.n_embd = -1;
            result.data.deinit(allocator);
            break;
        }
        const n_embd = @as(usize, @intCast(result.n_embd));
        const cvec_embd = @as(usize, @intCast(cvec.n_embd));
        const required_size = n_embd * @as(usize, (result.data.items.len / n_embd) + (cvec.data.items.len / cvec_embd));
        if (result.data.items.len < required_size) {
            try result.data.resize(allocator, required_size);
            @memset(result.data.items[result.data.items.len..required_size], 0.0);
        }

        const dst_start = n_embd * (result.data.items.len / n_embd - (cvec.data.items.len / cvec_embd));
        const dst = result.data.items[dst_start .. dst_start + cvec.data.items.len];

        for (dst, 0..) |*d, i| {
            d.* += cvec.data.items[i];
        }
    }

    return result;
}
pub fn common_control_vector_load_one(
    allocator: std.mem.Allocator,
    load_info: commonTypes.common_control_vector_load_info,
) !commonTypes.common_control_vector_data {
    var result = commonTypes.common_control_vector_data{
        .n_embd = -1,
        .data = .empty,
    };

    var ctx: ?*ggml.ggml_context = null;
    const meta_gguf_params = gguf.gguf_init_params{
        .no_alloc = false,
        .ctx = &ctx,
    };

    const fname_c = load_info.fname;
    defer allocator.free(fname_c);

    const ctx_gguf = gguf.gguf_init_from_file(fname_c.ptr, meta_gguf_params);
    if (ctx_gguf == null) {
        std.log.err("Failed to load control vector file from {s}", .{load_info.fname});
        return result;
    }
    defer gguf.gguf_free(ctx_gguf);

    const n_tensors = gguf.gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        std.log.warn("No direction tensors found in {s}", .{load_info.fname});
    }

    for (0..@as(usize, @intCast(n_tensors))) |i| {
        const name_ptr = gguf.gguf_get_tensor_name(ctx_gguf, @intCast(i));
        const name = std.mem.span(name_ptr);
        const dot_pos = std.mem.indexOfScalar(u8, name, '.');

        var layer_idx: i32 = -1;
        if (dot_pos) |pos| {
            const prefix = name[0..pos];
            if (std.mem.eql(u8, prefix, "direction")) {
                const suffix = name[pos + 1 ..];
                layer_idx = std.fmt.parseInt(i32, suffix, 10) catch -1;
            }
        }

        if (layer_idx < 0) {
            std.log.err("Invalid/unparsable direction tensor layer index in {s}", .{load_info.fname});
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            std.log.err("Invalid (zero) direction tensor layer index in {s}", .{load_info.fname});
            result.n_embd = -1;
            break;
        }

        const tensor = ggml.ggml_get_tensor(ctx.?, name_ptr);
        if (tensor == null) {
            std.log.err("Tensor {s} not found in context", .{name});
            result.n_embd = -1;
            break;
        }

        if (tensor.*.type != ggml.GGML_TYPE_F32) {
            std.log.err("Invalid (non-F32) direction tensor type in {s}", .{load_info.fname});
            result.n_embd = -1;
            break;
        }

        if (ggml.ggml_n_dims(tensor) != 1) {
            std.log.err("Invalid (non-1D) direction tensor shape in {s}", .{load_info.fname});
            result.n_embd = -1;
            break;
        }

        const nelements = ggml.ggml_nelements(tensor);
        if (result.n_embd == -1) {
            result.n_embd = @intCast(nelements);
        } else if (nelements != result.n_embd) {
            std.log.err("Direction tensor in {s} does not match previous dimensions", .{load_info.fname});
            result.n_embd = -1;
            break;
        }

        const embd: usize = @intCast(result.n_embd);
        const layer: usize = @intCast(layer_idx);

        const required_size = embd * layer;

        if (result.data.items.len < required_size) {
            const old_len = result.data.items.len;
            try result.data.resize(allocator, required_size);
            @memset(result.data.items[old_len..required_size], 0.0);
        }

        // Alignment-safe cast
        const src: [*]const f32 = @ptrCast(@alignCast(tensor.*.data));

        const dst_start = embd * (layer - 1);
        const dst = result.data.items[dst_start .. dst_start + embd];

        for (dst, 0..) |*d, j| {
            d.* += src[j] * load_info.strength;
        }
    }

    if (result.n_embd == -1) {
        std.log.warn("Skipping {s} due to invalid direction tensors", .{load_info.fname});
        result.data = .empty;
    }

    if (ctx) |c| {
        ggml.ggml_free(c);
    }

    return result;
}
