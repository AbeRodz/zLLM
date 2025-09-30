const std = @import("std");
const gguf = @import("gguf_converter.zig");
const registry = @import("../registry/model_registry.zig");
const registryRuntime = @import("../registry/runtime.zig");
const sampling = @import("llama_sampler.zig");
const RingBuffer = @import("../utils/ring_buffer.zig").RingBuffer;
pub const llama = @cImport({
    @cInclude("llama.h");
});
const client = @import("../client/client.zig");
const converter = @import("../safetensors/gguf/convert.zig");
const common = @import("llama_common.zig");

var message_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();

pub fn loadLlamaModelFromRegistry(model_name: []const u8, allocator: std.mem.Allocator) !*llama.struct_llama_model {
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
    const exists = try modelInfo.isCached();
    if (exists == false) {
        std.debug.print("Model not cached locally, downloading: {s}\n", .{model_name});
        client.downloader(modelInfo, allocator) catch |err| {
            std.debug.print("Error downloading model: {}\n", .{err});
            return err;
        };

        std.debug.print("Converting... \n", .{});
        try converter.convert(model_name, gguf_path.?, allocator);
    } else {
        std.debug.print("gguf_path{s}\n", .{gguf_path.?});
        const gguf_exists = try modelInfo.isGGUFCached();
        if (gguf_exists == false) {
            std.debug.print("Model found but gguf not cached, converting: {s}\n", .{model_name});
            try converter.convert(model_name, gguf_path.?, allocator);
        }
        std.debug.print("Model found in cache: {s}\n", .{model_name});
    }

    std.debug.print("loading gguf model: {s}\n", .{gguf_path.?});
    llama.llama_backend_init();

    var params = llama.llama_model_default_params();

    params.n_gpu_layers = 999;

    const model = llama.llama_model_load_from_file(gguf_path.?.ptr, params);
    if (model == null) {
        std.debug.print("Failed to load gguf model: {s}\n", .{gguf_path.?});
        return error.FailedToLoadModel;
    }
    std.debug.print("Model loaded successfully!\n", .{});
    return model.?;
}

fn appendMessage(
    allocator: std.mem.Allocator,
    //messages: *std.ArrayList(llama.struct_llama_chat_message),
    role: [*c]const u8,
    content: []const u8,
) !void {
    const dup = try allocator.dupeZ(u8, content);
    _ = message_ring.push(.{
        .role = role,
        .content = dup.ptr,
    });
}
fn applyChatTemplate(
    allocator: std.mem.Allocator,
    tmpl: [*c]const u8,
    //messages: std.ArrayList(llama.struct_llama_chat_message),
    messages: RingBuffer(llama.struct_llama_chat_message, 32),
    formatted: []u8,
) ![]u8 {
    var resized = try allocator.alloc(u8, 512);
    const c_messages: [*c]const llama.struct_llama_chat_message = @ptrCast(&messages.data[0]);
    var new_len = llama.llama_chat_apply_template(
        tmpl,
        c_messages,
        messages.count,
        true,
        formatted.ptr,
        @as(i32, @intCast(formatted.len)),
    );

    if (new_len < 0) {
        std.log.err("Failed to apply chat template\n", .{});
        return error.TemplateFailure;
    }

    if (@as(usize, @intCast(new_len)) > formatted.len) {
        resized = try allocator.alloc(u8, @as(usize, @intCast(new_len)));
        new_len = llama.llama_chat_apply_template(
            tmpl,
            c_messages,
            messages.count,
            true,
            resized.ptr,
            @as(i32, @intCast(formatted.len)),
        );
        return resized[0..@as(usize, @intCast(new_len))];
    }

    return formatted[0..@as(usize, @intCast(new_len))];
}

fn calculateBufferSize(n_ctx: u32, bytes_per_token: u32, headroom_percent: u32) usize {
    const base_size = n_ctx * bytes_per_token;
    return base_size + (base_size * headroom_percent / 100);
}
pub fn respondToPrompt(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    n_ctx: u32,
    prompt: []const u8,
) ![]u8 {
    const loaded = try registryRuntime.getOrLoadModel(allocator, model_name, 8192);
    const tmpl = llama.llama_model_chat_template(loaded.model, null);
    const sampler = llama_sampler();

    const bytes_per_token = 4;
    const headroom = 20;
    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);
    const backing_mem = try allocator.alloc(u8, buffer_size);
    defer allocator.free(backing_mem);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    const formatted = try allocator.alloc(u8, n_ctx);
    defer allocator.free(formatted);

    try appendMessage(allocator, "user", prompt);
    const chat_prompt = try applyChatTemplate(fast_alloc, tmpl, message_ring, formatted);

    var response_stream = std.ArrayList(u8).init(allocator);
    defer response_stream.deinit();
    const writer = response_stream.writer();

    const response = try generate(loaded.ctx, sampler, loaded.model, fast_alloc, chat_prompt, writer);
    try appendMessage(allocator, "assistant", response);

    return response_stream.toOwnedSlice();
}

pub fn execute(model_name: []const u8, n_ctx: u32, allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();

    const model = try loadLlamaModelFromRegistry(model_name, allocator);
    defer llama.llama_free_model(model);

    const ctx = try llama_context(model, n_ctx);
    defer llama.llama_free(ctx);

    const sampler = llama_sampler();
    const tmpl = llama.llama_model_chat_template(model, null);

    //var messages = std.ArrayList(llama.struct_llama_chat_message).init(allocator);
    //defer messages.deinit();

    var line_buf: [1024]u8 = undefined;

    const bytes_per_token = 4; // Could tune based on profiling
    const headroom = 20;

    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);
    const backing_mem = try allocator.alloc(u8, buffer_size);
    defer allocator.free(backing_mem);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    const formatted = try allocator.alloc(u8, n_ctx);
    defer allocator.free(formatted);

    // Greeting
    const init_prompt = "Please greet the user.";
    try appendMessage(allocator, "user", init_prompt);

    const greeting_prompt = try applyChatTemplate(fast_alloc, tmpl, message_ring, formatted);
    try stdout.print("\x1b[33m", .{});
    const greeting_response = try generate(ctx, sampler, model, fast_alloc, greeting_prompt, stdout);
    try stdout.print("\n\x1b[0m", .{});

    try appendMessage(allocator, "assistant", greeting_response);

    // Chat Loop
    while (true) {
        try stdout.print("\x1b[32m> \x1b[0m", .{});
        const input = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        if (input == null or input.?.len == 0) break;

        try appendMessage(allocator, "user", input.?);

        const prompt = try applyChatTemplate(fast_alloc, tmpl, message_ring, formatted);

        try stdout.print("\x1b[33m", .{});
        const response = try generate(ctx, sampler, model, fast_alloc, prompt, stdout);
        try stdout.print("\n\x1b[0m", .{});

        try appendMessage(allocator, "assistant", response);
        try stdout.print("{d}", .{message_ring.count});
        // Reset the allocator after each loop to reuse the same memory
        fixed_buffer_allocator.reset();
    }
}
pub const StreamIter = struct {
    ctx: *llama.struct_llama_context,
    //sampler: *sampling.CommonSampler,
    sampler: [*c]llama.struct_llama_sampler,
    model: *llama.struct_llama_model,
    allocator: std.mem.Allocator,
    vocab: *const llama.struct_llama_vocab,
    // token_cache: std.ArrayList(usize) = std.ArrayList(usize).init(std.heap.page_allocator),
    batch: llama.llama_batch,
    buf: [512]u8 = undefined,
    token_buf: [1]llama.llama_token = undefined,
    is_done: bool = false,
    prompt_token_count: i32,
    completion_token_count: i32 = 0,

    pub inline fn next(self: *StreamIter) !?[]const u8 {
        if (self.is_done) return error.EndOfStream;

        const n_ctx_used = llama.llama_kv_self_used_cells(self.ctx);
        if (n_ctx_used + self.batch.n_tokens > llama.llama_n_ctx(self.ctx)) {
            self.is_done = true;
            return error.EndOfStream;
        }

        if (llama.llama_decode(self.ctx, self.batch) != 0) {
            self.is_done = true;
            return error.EndOfStream;
        }

        const token = llama.llama_sampler_sample(self.sampler, self.ctx, -1);
        if (llama.llama_vocab_is_eog(self.vocab, token)) {
            self.is_done = true;
            return error.EndOfStream;
        }
        //var buf: [256]u8 = undefined;
        const len = llama.llama_token_to_piece(self.vocab, token, &self.buf, self.buf.len, 0, true);
        const lenCast = @as(usize, @intCast(len));
        if (len < 0 or lenCast > self.buf.len) {
            self.is_done = true;
            return error.InvalidTokenLength;
            //return null;
        }
        const slice = self.buf[0..lenCast];

        self.token_buf[0] = token;
        const tok = @as([*c]llama.llama_token, &self.token_buf);
        self.batch.token = tok;
        self.batch.n_tokens = 1;
        //const batch = llama.llama_batch_get_one(tok, 1);
        //self.batch = batch;

        self.completion_token_count += 1;
        return slice;
    }
    pub fn nextBatchedCommon(self: *StreamIter) !?[]const u8 {
        if (self.is_done) return null;

        const n_ctx_used = llama.llama_kv_self_used_cells(self.ctx);
        if (n_ctx_used + self.batch.n_tokens > llama.llama_n_ctx(self.ctx)) {
            self.is_done = true;
            return null;
        }
        // if (llama.llama_decode(self.ctx, self.batch) != 0) {
        //     self.is_done = true;
        //     return null;
        // }
        // Batch size for multiple tokens at once
        const batch_size = 8;
        var token_buf: [256]u8 = undefined; // max token piece length buffer
        var tokens: [batch_size]llama.llama_token = undefined; // tokens buffer

        var output = std.ArrayList(u8).init(std.heap.page_allocator);

        // Clear batch before adding tokens
        //common.common_batch_clear(&self.batch);

        for (0..batch_size) |i| {
            //const token = llama.llama_sampler_sample(self.sampler, self.ctx, -1);
            const token = sampling.common_sampler_sample(self.sampler, self.ctx, -1, true);
            sampling.common_sampler_accept(self.sampler, token, true);
            // Stop if end-of-generation token
            if (llama.llama_vocab_is_eog(self.vocab, token)) {
                self.is_done = true;
                break;
            }

            tokens[i] = token;

            // Add token to batch with common_batch_add
            // Arguments: (batch, tokens ptr, token count, seq_id, logits_pos, is_embd)
            // Using seq_id = i for example (distinct per token in batch)
            // logits_pos = 0 (starting logit position for this token)
            common.common_batch_add(
                &self.batch,
                token,
                @as(llama.llama_pos, @intCast(n_ctx_used)),
                &[_]i32{0}, // sequence id (unique per token)
                true, // logits offset
            );
        }

        if (self.batch.n_tokens == 0) {
            // no tokens added - end of generation
            output.deinit();
            return null;
        }

        // Run decoding on the batch of tokens
        if (llama.llama_decode(self.ctx, self.batch) != 0) {
            self.is_done = true;
            output.deinit();
            return null;
        }

        // For each token, convert to piece and append to output buffer
        for (0..@as(usize, @intCast(self.batch.n_tokens))) |i| {
            const len = llama.llama_token_to_piece(self.vocab, tokens[i], &token_buf, token_buf.len, 0, true);
            if (len < 0 or @as(usize, @intCast(len)) > token_buf.len) {
                std.debug.print("Invalid token piece length: {}\n", .{len});
                self.is_done = true;
                output.deinit();
                return null;
            }

            try output.appendSlice(token_buf[0..@as(usize, @intCast(len))]);
        }

        const result = try output.toOwnedSlice();
        output.deinit();

        return result;
    }

    pub fn deinit(_: *StreamIter) void {
        // No dynamic memory in struct currently might need in the future?
    }
};
fn createSeqIds(allocator: std.mem.Allocator, n_parallel: usize) ![]i32 {
    const seq_ids = try allocator.alloc(i32, n_parallel);
    for (seq_ids, 0..) |*id, i| {
        id.* = @as(i32, @intCast(i));
    }
    return seq_ids;
}

pub fn respondToPromptStream(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    n_ctx: u32,
    prompt: []const u8,
) !StreamIter {
    const loaded = try registryRuntime.getOrLoadModel(allocator, model_name, n_ctx);
    const tmpl = llama.llama_model_chat_template(loaded.model, null);
    const sampler = llama_sampler();
    // const params = common.CommonParams{
    //     .sampling = .{ .temp = 0.2 },
    // };
    //const sampler = try sampling.CommonSampler.init(allocator, loaded.model, params.sampling);
    const bytes_per_token = 4;
    const headroom = 20;
    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);
    const backing_mem = try allocator.alloc(u8, buffer_size);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    const formatted = try allocator.alloc(u8, n_ctx);
    try appendMessage(allocator, "user", prompt);
    const chat_prompt = try applyChatTemplate(fast_alloc, tmpl, message_ring, formatted);

    const vocab = llama.llama_model_get_vocab(loaded.model);
    const is_first = llama.llama_kv_self_used_cells(loaded.ctx) == 0;
    const n_prompt = -llama.llama_tokenize(vocab, chat_prompt.ptr, @as(i32, @intCast(chat_prompt.len)), null, 0, is_first, true);

    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    if (llama.llama_tokenize(vocab, chat_prompt.ptr, @as(i32, @intCast(chat_prompt.len)), prompt_tokens.ptr, @as(i32, @intCast(chat_prompt.len)), is_first, true) < 0) {
        return error.TokenizationFailed;
    }
    //const batch = llama.llama_batch_init(n_prompt, 0, 1);
    const cpu = try std.Thread.getCpuCount();
    const batches = @as(i32, @intCast(@divFloor(cpu, 4)));
    const batch = llama.llama_batch_get_one(prompt_tokens.ptr, n_prompt);
    llama.llama_set_n_threads(loaded.ctx, @as(i32, @intCast(cpu)), batches);
    return StreamIter{
        .ctx = loaded.ctx,
        //.sampler = sampler,
        .sampler = sampler,
        .model = loaded.model,
        .allocator = allocator,
        .vocab = vocab.?,
        .batch = batch,
        .prompt_token_count = n_prompt,
    };
}

fn generate(
    ctx: *llama.struct_llama_context,
    smpl: [*c]llama.struct_llama_sampler,
    model: *llama.struct_llama_model,
    allocator: std.mem.Allocator,
    prompt: []const u8,
    writer: anytype,
) ![]u8 {
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
        if (n_ctx_used + batch.n_tokens > llama.llama_n_ctx(ctx)) break;

        if (llama.llama_decode(ctx, batch) != 0) return error.DecodeFailed;

        new_token_id = llama.llama_sampler_sample(smpl, ctx, -1);
        if (llama.llama_vocab_is_eog(vocab, new_token_id)) break;

        var buf: [4096]u8 = undefined;
        const len = llama.llama_token_to_piece(vocab, new_token_id, &buf, buf.len, 0, true);
        if (len < 0) return error.TokenToPieceFailed;

        const slice = buf[0..@as(usize, @intCast(len))];
        try response.appendSlice(slice);
        try writer.print("{s}", .{slice});

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
