const std = @import("std");
const gguf = @import("gguf_converter.zig");
const registry = @import("../registry/model_registry.zig");
const registryRuntime = @import("../registry/runtime.zig");
const RingBuffer = @import("../utils/ring_buffer.zig").RingBuffer;
pub const llama = @cImport({
    @cInclude("llama.h");
});

var message_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();

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

    const model = try loadLlamaModelFromRegistry(model_name);
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
    sampler: [*c]llama.struct_llama_sampler,
    model: *llama.struct_llama_model,
    allocator: std.mem.Allocator,
    vocab: *const llama.struct_llama_vocab,
    batch: llama.llama_batch,
    is_done: bool = false,
    prompt_token_count: i32,
    completion_token_count: i32 = 0,

    pub fn next(self: *StreamIter) !?[]const u8 {
        if (self.is_done) return null;

        const n_ctx_used = llama.llama_kv_self_used_cells(self.ctx);
        if (n_ctx_used + self.batch.n_tokens > llama.llama_n_ctx(self.ctx)) {
            self.is_done = true;
            return null;
        }

        if (llama.llama_decode(self.ctx, self.batch) != 0) {
            self.is_done = true;
            return null;
        }

        const token = llama.llama_sampler_sample(self.sampler, self.ctx, -1);
        if (llama.llama_vocab_is_eog(self.vocab, token)) {
            self.is_done = true;
            return null;
        }
        var buf: [4096]u8 = undefined;
        const len = llama.llama_token_to_piece(self.vocab, token, &buf, buf.len, 0, true);
        if (len < 0 or @as(usize, @intCast(len)) > buf.len) {
            std.debug.print("Invalid len = {}\n", .{len});
            self.is_done = true;
            return null;
        }
        const slice = buf[0..@as(usize, @intCast(len))];
        // const slice = self.allocator.dupe(u8, buf[0..@as(usize, @intCast(len))]) catch {
        //     self.is_done = true;
        //     return null;
        // };
        //const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
        const tokens = try std.heap.page_allocator.alloc(i32, 1);
        tokens[0] = token;
        const tok = @as([*c]i32, @ptrCast(tokens.ptr));
        const batch = llama.llama_batch_get_one(tok, 1);
        self.batch = batch;
        //defer std.heap.page_allocator.free(tokens);
        self.completion_token_count += 1;
        return slice;
    }

    pub fn deinit(_: *StreamIter) void {
        // No dynamic memory in struct currently might need in the future?
    }
};

fn generateStream(
    ctx: *llama.struct_llama_context,
    smpl: [*c]llama.struct_llama_sampler,
    model: *llama.struct_llama_model,
    allocator: std.mem.Allocator,
    prompt: []const u8,
    on_token: fn ([]const u8) anyerror!void,
) !void {
    const vocab = llama.llama_model_get_vocab(model);
    const is_first = llama.llama_kv_self_used_cells(ctx) == 0;
    const n_prompt = -llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), null, 0, is_first, true);

    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    defer allocator.free(prompt_tokens);

    if (llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), prompt_tokens.ptr, @as(i32, @intCast(prompt.len)), is_first, true) < 0) {
        return error.TokenizationFailed;
    }

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
        try on_token(slice);

        batch = llama.llama_batch_get_one(&new_token_id, 1);
    }
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

    const batch = llama.llama_batch_get_one(prompt_tokens.ptr, n_prompt);

    return StreamIter{
        .ctx = loaded.ctx,
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
