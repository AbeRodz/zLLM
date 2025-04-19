const std = @import("std");
const gguf = @import("gguf_converter.zig");
const registry = @import("../registry/model_registry.zig");

pub const llama = @cImport({
    @cInclude("llama.h");
});

pub fn loadLlamaModelFromRegistry(model_name: []const u8) !*llama.struct_llama_model {
    const modelInfo = try registry.findModelErrorless(model_name) orelse return error.UnknownModel;

    var gguf_path: ?[]const u8 = null;
    for (modelInfo.files) |file| {
        if (std.mem.endsWith(u8, file, ".gguf")) {
            gguf_path = try modelInfo.localFilePath(modelInfo.name, file);
            break;
        }
    }

    if (gguf_path == null) {
        gguf_path = try modelInfo.localFilePath(modelInfo.name, "model.gguf");
    }
    llama.ggml_backend_load_all();

    var params = llama.llama_model_default_params();
    const n_gpu_layers = 99;
    params.n_gpu_layers = n_gpu_layers;

    const model = llama.llama_model_load_from_file(gguf_path.?.ptr, params);
    if (model == null) {
        std.debug.print("Failed to load gguf model: {s}\n", .{gguf_path.?});
        return error.FailedToLoadModel;
    }
    return model.?;
}

pub fn execute(user_input: []const u8, model_name: []const u8, n_ctx: u32, allocator: std.mem.Allocator) !void {
    const model = try loadLlamaModelFromRegistry(model_name);
    defer llama.llama_free_model(model);
    const ctx = try llama_context(model, n_ctx);
    defer llama.llama_free(ctx);
    const sampler = llama_sampler();
    const tmpl = llama.llama_model_chat_template(model, null);
    var messages = std.ArrayList(llama.struct_llama_chat_message).init(allocator);
    defer messages.deinit();

    const max_buf_size = n_ctx;
    var formatted = try allocator.alloc(u8, max_buf_size);
    defer allocator.free(formatted);

    var prev_len: usize = 0;
    while (true) {
        std.debug.print("\x1b[32m> \x1b[0m", .{});

        if (user_input.len == 0) break;

        const user_input_dup = try allocator.dupeZ(u8, user_input);

        // Add user message
        try messages.append(.{
            .role = "user",
            .content = user_input_dup.ptr,
        });

        // Apply template
        var new_len = llama.llama_chat_apply_template(tmpl, messages.items.ptr, messages.items.len, true, formatted.ptr, @as(i32, @intCast(formatted.len)));
        if (new_len < 0) {
            std.debug.print("Failed to apply chat template\n", .{});
            return;
        }

        // If too small, resize and try again
        if (@as(usize, @intCast(new_len)) > formatted.len) {
            formatted = try allocator.realloc(formatted, @as(usize, @intCast(new_len)));
            new_len = llama.llama_chat_apply_template(tmpl, messages.items.ptr, messages.items.len, true, formatted.ptr, @as(i32, @intCast(formatted.len)));
        }

        const prompt = formatted[prev_len..@as(usize, @intCast(new_len))];

        std.debug.print("\x1b[33m", .{});
        const response = try generate(ctx, sampler, model, allocator, prompt);
        std.debug.print("\n\x1b[0m", .{});

        const resp_dup = try allocator.dupeZ(u8, response);
        try messages.append(.{
            .role = "assistant",
            .content = resp_dup.ptr,
        });

        prev_len = @as(usize, @intCast(new_len));
    }
}

fn generate(ctx: *llama.struct_llama_context, smpl: [*c]llama.struct_llama_sampler, model: *llama.struct_llama_model, allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    const vocab = llama.llama_model_get_vocab(model);

    const is_first = llama.llama_kv_self_used_cells(ctx) == 0;
    const n_prompt = -llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), null, 0, is_first, true);
    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    defer allocator.free(prompt_tokens);

    if (llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), prompt_tokens.ptr, @as(i32, @intCast(prompt.len)), is_first, true) < 0) {
        return error.TokenizationFailed;
    }

    var response = std.ArrayList(u8).init(allocator);
    defer response.deinit();

    var batch = llama.llama_batch_get_one(prompt_tokens.ptr, n_prompt);
    var new_token_id: llama.llama_token = undefined;

    while (true) {
        const n_ctx_used = llama.llama_kv_self_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > llama.llama_n_ctx(ctx)) {
            break;
        }

        if (llama.llama_decode(ctx, batch) != 0) {
            return error.DecodeFailed;
        }

        new_token_id = llama.llama_sampler_sample(smpl, ctx, -1);
        if (llama.llama_vocab_is_eog(vocab, new_token_id)) break;

        var buf: [256]u8 = undefined;
        const len = llama.llama_token_to_piece(vocab, new_token_id, &buf, buf.len, 0, true);
        if (len < 0) return error.TokenToPieceFailed;

        try response.appendSlice(buf[0..@as(usize, @intCast(len))]);
        std.debug.print("{s}", .{buf[0..@as(usize, @intCast(len))]});
        batch = llama.llama_batch_get_one(&new_token_id, 1);
    }

    return response.toOwnedSlice();
}

pub const Tokenize = struct { vocab: *const llama.struct_llama_vocab, n_prompt: i32, prompt_tokens: []i32 };

pub fn tokenize(model: *llama.struct_llama_model, allocator: std.mem.Allocator, prompt: []const u8) !Tokenize {
    const vocab = llama.llama_model_get_vocab(model);

    const n_prompt = -llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), null, 0, true, true);
    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    defer allocator.free(prompt_tokens);
    if (llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), prompt_tokens.ptr, @as(i32, @intCast(prompt.len)), true, true) < 0) {
        return error.TokenizationFailed;
    }
    return Tokenize{
        .vocab = vocab.?,
        .n_prompt = n_prompt,
        .prompt_tokens = prompt_tokens,
    };
}

pub fn llama_context(model: *llama.struct_llama_model, n_ctx: u32) !*llama.struct_llama_context {
    var ctx_params = llama.llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    const ctx = llama.llama_init_from_model(model, ctx_params);
    if (ctx == null) {
        std.debug.print("Failed to create llama context", .{});
        return error.FailedToContextCreation;
    }
    return ctx.?;
}

pub fn llama_sampler() [*c]llama.struct_llama_sampler {
    const smpl = llama.llama_sampler_chain_init(llama.llama_sampler_chain_default_params());
    llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_greedy());
    llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_min_p(0.05, 1));
    llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_temp(0.8));
    llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_dist(llama.LLAMA_DEFAULT_SEED));
    return smpl;
}
