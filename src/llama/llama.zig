const std = @import("std");
const gguf = @import("gguf_converter.zig");
const registry = @import("../registry/model_registry.zig");
const registryRuntime = @import("../registry/runtime.zig");
const sampling = @import("llama_sampler.zig");
const RingBuffer = @import("../utils/ring_buffer.zig").RingBuffer;
const session_mgr = @import("../inference/session_manager.zig");
pub const chat_format = @import("chat_format.zig");

pub const llama = @cImport({
    @cInclude("llama.h");
});
const llama_model = @import("cTypes.zig");
const client = @import("../client/client.zig");
const converter = @import("../safetensors/gguf/convert.zig");
const common = @import("llama_common.zig");

// NOTE: message_ring is no longer a global.  Each call site creates a local
// RingBuffer on the stack and passes it explicitly.  This eliminates the
// shared-state race that corrupted concurrent requests.

pub fn loadLlamaModelFromRegistry(model_name: []const u8, allocator: std.mem.Allocator) !*llama_model.LlamaModel {
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

    var params = llama_model.default_params();
    params.n_gpu_layers = 999;

    const model = llama_model.loadModel(gguf_path.?, params);
    if (model == null) {
        std.debug.print("Failed to load gguf model: {s}\n", .{gguf_path.?});
        return error.FailedToLoadModel;
    }
    std.debug.print("Model loaded successfully!\n", .{});
    return model.?;
}

// ---------------------------------------------------------------------------
// Chat template helpers
// ---------------------------------------------------------------------------

/// Push a message onto a caller-owned ring buffer.
/// The content string is duped into allocator so the ring entry is stable.
fn appendMessage(
    allocator: std.mem.Allocator,
    ring: *RingBuffer(llama.struct_llama_chat_message, 32),
    role: [*c]const u8,
    content: []const u8,
) !void {
    const dup = try allocator.dupeZ(u8, content);
    if (!ring.push(.{ .role = role, .content = dup.ptr })) {
        std.log.warn("message ring full, dropping message", .{});
    }
}

fn applyChatTemplate(
    allocator: std.mem.Allocator,
    tmpl: [*c]const u8,
    messages: *RingBuffer(llama.struct_llama_chat_message, 32),
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
            @as(i32, @intCast(resized.len)),
        );
        return resized[0..@as(usize, @intCast(new_len))];
    }

    return formatted[0..@as(usize, @intCast(new_len))];
}

fn calculateBufferSize(n_ctx: u32, bytes_per_token: u32, headroom_percent: u32) usize {
    const base_size = n_ctx * bytes_per_token;
    return base_size + (base_size * headroom_percent / 100);
}

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Structured message passed from the API layer.
pub const ApiMessage = struct {
    role: []const u8,
    content: []const u8,
};

/// Returned by respondToPrompt; carries the generated text and real token counts.
pub const PromptResult = struct {
    content: []u8,
    prompt_tokens: i32,
    completion_tokens: i32,
    /// Non-null when the model output was parsed as a tool call.
    /// Only populated by respondToPrompt (non-streaming path).
    tool_call: ?chat_format.ParsedToolCall = null,
};

/// Hard deadline for a single inference call.
/// Streaming: checked at the start of every StreamIter.next() call.
/// Non-streaming: checked at the start of every generate() loop iteration.
/// Pass std.math.maxInt(i128) to disable (CLI interactive mode).
pub const INFERENCE_TIMEOUT_NS: i128 = 120 * std.time.ns_per_s;

// ---------------------------------------------------------------------------
// Non-streaming inference
// ---------------------------------------------------------------------------

pub fn respondToPrompt(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    n_ctx: u32,
    messages: []const ApiMessage,
    /// Optional X-Session-ID header value.  When provided the call borrows an
    /// existing llama_context from the session pool instead of allocating one
    /// per request, eliminating the Metal KV-cache alloc/free overhead (~50ms).
    session_id: ?[]const u8,
    /// Tool definitions for this request.  When non-null a GBNF lazy grammar
    /// sampler is added to constrain output to a valid tool call JSON once the
    /// trigger token is emitted.  Pass null for regular (non-tool) requests.
    tools: ?[]const chat_format.ApiTool,
) !PromptResult {
    const loaded = try registryRuntime.getOrLoadModel(allocator, model_name);
    const tmpl = llama.llama_model_chat_template(@ptrCast(loaded.model), null);
    const family = chat_format.detect(tmpl);

    // Build the appropriate sampler: with lazy grammar when tools are present,
    // plain greedy otherwise.
    const sampler = if (tools != null and tools.?.len > 0) blk: {
        const vocab = llama.llama_model_get_vocab(@ptrCast(loaded.model));
        const grammar_z = chat_format.buildGrammar(allocator, tools.?) catch |err| {
            std.log.warn("grammar build failed ({s}), falling back to greedy sampler", .{@errorName(err)});
            break :blk llama_sampler();
        };
        defer allocator.free(grammar_z);
        break :blk llamaSamplerWithGrammar(vocab.?, grammar_z, chat_format.triggerPattern(family));
    } else llama_sampler();
    defer llama.llama_sampler_free(sampler);

    // ------------------------------------------------------------------
    // Resolve context: borrow from session pool or allocate fresh
    // ------------------------------------------------------------------
    var session: ?*session_mgr.Session = null;
    var owns_ctx = false;

    const ctx: *llama.struct_llama_context = blk: {
        if (session_id) |sid| {
            const s = session_mgr.acquire(sid, @ptrCast(loaded.model)) catch |err| switch (err) {
                error.SessionBusy,
                error.NoSlotAvailable,
                error.ManagerNotInitialized,
                => {
                    std.log.warn("session acquire failed ({s}), using stateless ctx", .{@errorName(err)});
                    owns_ctx = true;
                    break :blk try llama_context(@ptrCast(loaded.model), n_ctx);
                },
                else => return err,
            };
            session = s;
            break :blk @ptrCast(s.ctx);
        } else {
            owns_ctx = true;
            break :blk try llama_context(@ptrCast(loaded.model), n_ctx);
        }
    };
    defer {
        if (owns_ctx) llama.llama_free(ctx);
        if (session) |s| session_mgr.release(s);
    }

    // FBA only used by applyChatTemplate for its retry buffer (≤ n_ctx bytes).
    const backing_mem = try allocator.alloc(u8, n_ctx);
    defer allocator.free(backing_mem);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    const formatted = try allocator.alloc(u8, n_ctx);
    defer allocator.free(formatted);

    var local_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();
    for (messages) |msg| {
        try appendMessage(allocator, &local_ring, msg.role.ptr, msg.content);
    }
    const chat_prompt = try applyChatTemplate(fast_alloc, tmpl, &local_ring, formatted);

    const deadline = std.time.nanoTimestamp() + INFERENCE_TIMEOUT_NS;
    var result = try generate(ctx, sampler, @ptrCast(loaded.model), allocator, chat_prompt, null, deadline);

    // Parse tool call from result when tools were declared.
    if (tools != null and tools.?.len > 0) {
        result.tool_call = chat_format.parseOutput(allocator, family, result.content);
    }

    return result;
}

// ---------------------------------------------------------------------------
// CLI interactive mode
// ---------------------------------------------------------------------------

pub fn execute_v2(model_name: []const u8, n_ctx: u32, allocator: std.mem.Allocator) !void {
    var stdout_buffer: [4028]u8 = undefined;
    var stdin_buffer: [4028]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    var stdout = &stdout_writer.interface;

    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
    const stdin = &stdin_reader.interface;

    const model = try loadLlamaModelFromRegistry(model_name, allocator);
    const model_ptr: ?*llama.struct_llama_model = @ptrCast(model);
    defer llama.llama_free_model(model_ptr);

    const ctx = try llama_context(model, n_ctx);
    defer llama.llama_free(ctx);

    const sampler = llama_sampler();
    defer llama.llama_sampler_free(sampler);

    const tmpl = llama.llama_model_chat_template(model_ptr, null);

    const bytes_per_token = 4;
    const headroom = 100;
    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);

    const backing_mem = try allocator.alloc(u8, buffer_size);
    defer allocator.free(backing_mem);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);

    var local_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();

    while (true) {
        const input = stdin.takeDelimiterExclusive('\n') catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (input.len == 0) continue;

        const user_copy = try allocator.alloc(u8, input.len);
        @memcpy(user_copy, input);
        try appendMessage(allocator, &local_ring, "user", user_copy);

        const formatted = try allocator.alloc(u8, buffer_size);
        const prompt = try applyChatTemplate(allocator, tmpl, &local_ring, formatted);

        const result = try generate(ctx, sampler, @ptrCast(model), allocator, prompt, stdout, std.math.maxInt(i128));

        const assistant_copy = try allocator.alloc(u8, result.content.len);
        @memcpy(assistant_copy, result.content);
        try appendMessage(allocator, &local_ring, "assistant", assistant_copy);

        try stdout.print("\n", .{});
        try stdout.flush();

        fixed_buffer_allocator.reset();
    }
}

pub fn execute(model_name: []const u8, n_ctx: u32, allocator: std.mem.Allocator) !void {
    var stdout_buffer: [4028]u8 = undefined;
    var stdin_buffer: [4028]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    var stdout = &stdout_writer.interface;

    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
    const stdin = &stdin_reader.interface;

    const model = try loadLlamaModelFromRegistry(model_name, allocator);
    const model_ptr: ?*llama.struct_llama_model = @ptrCast(model);
    defer llama.llama_free_model(model_ptr);

    const ctx = try llama_context(model, n_ctx);
    defer llama.llama_free(ctx);

    const sampler = llama_sampler();
    defer llama.llama_sampler_free(sampler);

    const tmpl = llama.llama_model_chat_template(model_ptr, null);

    const bytes_per_token = 4;
    const headroom = 20;
    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);

    const backing_mem = try allocator.alloc(u8, buffer_size * 100);
    defer allocator.free(backing_mem);
    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    var local_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();

    while (stdin.takeDelimiterExclusive('\n')) |input| {
        stdin.toss(1);
        if (input.len == 0) continue;

        try appendMessage(allocator, &local_ring, "user", input);

        const formatted = try fast_alloc.alloc(u8, buffer_size);
        defer fast_alloc.free(formatted);
        const prompt = try applyChatTemplate(fast_alloc, tmpl, &local_ring, formatted);

        try stdout.print("\x1b[33m", .{});
        const result = try generate(ctx, sampler, @ptrCast(model), fast_alloc, prompt, stdout, std.math.maxInt(i128));

        try appendMessage(allocator, &local_ring, "assistant", result.content);

        try stdout.print("\n", .{});
        try stdout.flush();
        fixed_buffer_allocator.reset();
    } else |err| switch (err) {
        error.EndOfStream => {},
        error.StreamTooLong => return err,
        error.ReadFailed => return err,
    }
}

/// Single-shot generation: apply a chat template to `prompt`, generate tokens,
/// stream them to stdout, then print timing stats identical to run-lookahead.
pub fn execute_prompt(model_name: []const u8, prompt: []const u8, n_ctx: u32, allocator: std.mem.Allocator) !void {
    var stdout_buffer: [4028]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const model = try loadLlamaModelFromRegistry(model_name, allocator);
    const model_ptr: ?*llama.struct_llama_model = @ptrCast(model);
    defer llama.llama_free_model(model_ptr);

    const ctx = try llama_context(model, n_ctx);
    defer llama.llama_free(ctx);

    const sampler = llama_sampler();
    defer llama.llama_sampler_free(sampler);

    const tmpl = llama.llama_model_chat_template(model_ptr, null);

    const bytes_per_token = 4;
    const headroom = 20;
    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);

    const backing_mem = try allocator.alloc(u8, buffer_size * 100);
    defer allocator.free(backing_mem);
    var fba = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fba.allocator();

    var ring = RingBuffer(llama.struct_llama_chat_message, 32).init();
    try appendMessage(allocator, &ring, "user", prompt);

    const formatted = try fast_alloc.alloc(u8, buffer_size);
    const full_prompt = try applyChatTemplate(fast_alloc, tmpl, &ring, formatted);

    const t_start = std.time.nanoTimestamp();

    try stdout.print("\x1b[33m", .{});
    const result = try generate(ctx, sampler, @ptrCast(model), fast_alloc, full_prompt, stdout, std.math.maxInt(i128));
    try stdout.print("\x1b[0m\n\n", .{});
    try stdout.flush();

    const t_end = std.time.nanoTimestamp();
    const elapsed_s = @as(f64, @floatFromInt(t_end - t_start)) / 1e9;

    std.debug.print("decoded {d} tokens in {d:.3} s, speed: {d:.3} t/s (greedy)\n", .{
        result.completion_tokens,
        elapsed_s,
        @as(f64, @floatFromInt(result.completion_tokens)) / elapsed_s,
    });
}

pub fn execute_og(model_name: []const u8, n_ctx: u32, allocator: std.mem.Allocator) !void {
    var stdout_buffer: [4028]u8 = undefined;
    var stdin_buffer: [4028]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    var stdout = &stdout_writer.interface;

    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
    const stdin = &stdin_reader.interface;

    const model = try loadLlamaModelFromRegistry(model_name, allocator);
    const model_ptr: ?*llama.struct_llama_model = @ptrCast(model);
    defer llama.llama_free_model(model_ptr);

    const ctx = try llama_context(model, n_ctx);
    defer llama.llama_free(ctx);

    const sampler = llama_sampler();
    defer llama.llama_sampler_free(sampler);

    const tmpl = llama.llama_model_chat_template(model_ptr, null);

    const bytes_per_token = 4;
    const headroom = 20;
    const buffer_size = calculateBufferSize(n_ctx, bytes_per_token, headroom);

    const backing_mem = try allocator.alloc(u8, buffer_size);
    defer allocator.free(backing_mem);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    const formatted = try fast_alloc.alloc(u8, buffer_size);

    var local_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();

    while (true) {
        const input = stdin.takeDelimiterExclusive('\n') catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (input.len == 0) continue;

        try appendMessage(allocator, &local_ring, "user", input);
        const prompt = try applyChatTemplate(fast_alloc, tmpl, &local_ring, formatted);

        const result = try generate(ctx, sampler, @ptrCast(model), fast_alloc, prompt, stdout);
        try appendMessage(allocator, &local_ring, "assistant", result.content);
        try stdout.print("\n", .{});
        try stdout.flush();
        fixed_buffer_allocator.reset();
    }
}

// ---------------------------------------------------------------------------
// Streaming iterator
// ---------------------------------------------------------------------------

pub const StreamIter = struct {
    ctx: *llama.struct_llama_context,
    sampler: [*c]llama.struct_llama_sampler,
    model: *llama_model.LlamaModel,
    allocator: std.mem.Allocator,
    vocab: *const llama.struct_llama_vocab,
    batch: llama.llama_batch,
    buf: [512]u8 = undefined,
    token_buf: [1]llama.llama_token = undefined,
    is_done: bool = false,
    prompt_token_count: i32,
    completion_token_count: i32 = 0,
    /// Absolute nanosecond deadline; next() returns InferenceTimeout if exceeded.
    deadline_ns: i128,
    /// True when this iterator created its own context (stateless path).
    /// The context is freed in deinit().
    owns_ctx: bool,
    /// Non-null when inference runs inside a session (session path).
    /// deinit() releases the session back to the manager.
    session: ?*session_mgr.Session,

    pub inline fn next(self: *StreamIter) !?[]const u8 {
        if (self.is_done) return error.EndOfStream;
        if (std.time.nanoTimestamp() > self.deadline_ns) {
            self.is_done = true;
            return error.InferenceTimeout;
        }

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

        const len = llama.llama_token_to_piece(self.vocab, token, &self.buf, self.buf.len, 0, true);
        const lenCast = @as(usize, @intCast(len));
        if (len < 0 or lenCast > self.buf.len) {
            self.is_done = true;
            return error.InvalidTokenLength;
        }
        const slice = self.buf[0..lenCast];

        self.token_buf[0] = token;
        const tok = @as([*c]llama.llama_token, &self.token_buf);
        self.batch.token = tok;
        self.batch.n_tokens = 1;

        self.completion_token_count += 1;
        return slice;
    }

    pub fn deinit(self: *StreamIter) void {
        llama.llama_sampler_free(self.sampler);
        // Free the context only when we own it (stateless path).
        if (self.owns_ctx) {
            llama.llama_free(self.ctx);
        }
        // Return the session to the pool (session path).
        if (self.session) |s| {
            session_mgr.release(s);
        }
    }
};

// ---------------------------------------------------------------------------
// Streaming inference — session-aware
// ---------------------------------------------------------------------------

/// Build a streaming iterator for the given request.
///
/// session_id (optional)
///   When provided, the call attempts to acquire a matching session from the
///   global SessionManager.  On success, the session's llama_context is reused
///   and released in StreamIter.deinit().
///
///   If the session is busy or the slot cap is exhausted, the call falls back
///   to creating a temporary context (owns_ctx = true) so the request is never
///   dropped.
///
///   When session_id is null the stateless path is always used.
pub fn respondToPromptStream(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    n_ctx: u32,
    messages: []const ApiMessage,
    session_id: ?[]const u8,
    /// Tool definitions — when non-null a lazy GBNF grammar sampler is inserted
    /// before the greedy sampler so the JSON after the trigger token is valid.
    tools: ?[]const chat_format.ApiTool,
) !StreamIter {
    const loaded = try registryRuntime.getOrLoadModel(allocator, model_name);
    const tmpl = llama.llama_model_chat_template(@ptrCast(loaded.model), null);
    const family = chat_format.detect(tmpl);

    const sampler = if (tools != null and tools.?.len > 0) blk: {
        const vocab = llama.llama_model_get_vocab(@ptrCast(loaded.model));
        const grammar_z = chat_format.buildGrammar(allocator, tools.?) catch |err| {
            std.log.warn("grammar build failed ({s}), falling back to greedy sampler", .{@errorName(err)});
            break :blk llama_sampler();
        };
        defer allocator.free(grammar_z);
        break :blk llamaSamplerWithGrammar(vocab.?, grammar_z, chat_format.triggerPattern(family));
    } else llama_sampler();

    // ------------------------------------------------------------------
    // Resolve context: session-owned or freshly created (stateless)
    // ------------------------------------------------------------------
    var session: ?*session_mgr.Session = null;
    var owns_ctx = false;

    const ctx: *llama.struct_llama_context = blk: {
        if (session_id) |sid| {
            const s = session_mgr.acquire(sid, @ptrCast(loaded.model)) catch |err| switch (err) {
                // Graceful degradation — fall back to a fresh stateless context.
                error.SessionBusy,
                error.NoSlotAvailable,
                error.ManagerNotInitialized,
                => {
                    std.log.warn("session acquire failed ({s}), falling back to stateless ctx", .{@errorName(err)});
                    owns_ctx = true;
                    const new_ctx = try llama_context(@ptrCast(loaded.model), n_ctx);
                    break :blk new_ctx;
                },
                else => return err,
            };
            session = s;
            // KV already cleared inside session_mgr.acquire().
            break :blk @ptrCast(s.ctx);
        } else {
            owns_ctx = true;
            break :blk try llama_context(@ptrCast(loaded.model), n_ctx);
        }
    };

    // For the stateless path the context is brand new; no clear needed.
    // For the session path, session_mgr.acquire() already called llama_kv_self_clear().

    // ------------------------------------------------------------------
    // Build prompt using a local ring — no shared global state
    // ------------------------------------------------------------------
    // The FixedBufferAllocator is only used by applyChatTemplate for its
    // internal retry buffer (≤ n_ctx bytes).  The old formula allocated
    // n_ctx*bytes_per_token*1.2 ≈ 40 KB; n_ctx bytes is enough.
    const backing_mem = try allocator.alloc(u8, n_ctx);

    var fixed_buffer_allocator = std.heap.FixedBufferAllocator.init(backing_mem);
    const fast_alloc = fixed_buffer_allocator.allocator();

    const formatted = try allocator.alloc(u8, n_ctx);

    var local_ring = RingBuffer(llama.struct_llama_chat_message, 32).init();
    for (messages) |msg| {
        try appendMessage(allocator, &local_ring, msg.role.ptr, msg.content);
    }
    const chat_prompt = try applyChatTemplate(fast_alloc, tmpl, &local_ring, formatted);

    // ------------------------------------------------------------------
    // Tokenize
    // ------------------------------------------------------------------
    const vocab = llama.llama_model_get_vocab(@ptrCast(loaded.model));

    const n_prompt = -llama.llama_tokenize(
        vocab,
        chat_prompt.ptr,
        @as(i32, @intCast(chat_prompt.len)),
        null,
        0,
        true,
        true,
    );

    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    if (llama.llama_tokenize(
        vocab,
        chat_prompt.ptr,
        @as(i32, @intCast(chat_prompt.len)),
        prompt_tokens.ptr,
        @as(i32, @intCast(prompt_tokens.len)),
        true,
        true,
    ) < 0) {
        return error.TokenizationFailed;
    }

    const batch = llama.llama_batch_get_one(prompt_tokens.ptr, n_prompt);

    return StreamIter{
        .ctx = ctx,
        .sampler = @ptrCast(sampler),
        .model = @ptrCast(loaded.model),
        .allocator = allocator,
        .vocab = vocab.?,
        .batch = batch,
        .prompt_token_count = n_prompt,
        .deadline_ns = std.time.nanoTimestamp() + INFERENCE_TIMEOUT_NS,
        .owns_ctx = owns_ctx,
        .session = session,
    };
}

// ---------------------------------------------------------------------------
// Core generation loop (non-streaming)
// ---------------------------------------------------------------------------

fn generate(
    ctx: *llama.struct_llama_context,
    smpl: [*c]llama.struct_llama_sampler,
    model: *llama.struct_llama_model,
    allocator: std.mem.Allocator,
    prompt: []const u8,
    /// Pass a real writer for CLI interactive mode (tokens streamed to stdout).
    /// Pass null for the HTTP non-streaming path — tokens are collected via
    /// the PromptResult return value; no writer allocation needed.
    writer: ?*std.io.Writer,
    /// Absolute nanosecond deadline from std.time.nanoTimestamp().
    /// Use std.math.maxInt(i128) to disable (CLI interactive mode).
    deadline_ns: i128,
) !PromptResult {
    const vocab = llama.llama_model_get_vocab(model);

    const is_first = llama.llama_kv_self_used_cells(ctx) == 0;
    const n_prompt = -llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), null, 0, is_first, true);
    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    defer allocator.free(prompt_tokens);

    if (llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), prompt_tokens.ptr, @as(i32, @intCast(prompt_tokens.len)), is_first, true) < 0) {
        return error.TokenizationFailed;
    }

    var response: std.ArrayList(u8) = .empty;
    defer response.deinit(allocator);

    var batch = llama.llama_batch_get_one(prompt_tokens.ptr, n_prompt);
    var new_token_id: llama.llama_token = undefined;
    var completion_tokens: i32 = 0;

    while (true) {
        if (std.time.nanoTimestamp() > deadline_ns) return error.InferenceTimeout;

        const n_ctx_used = llama.llama_kv_self_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > llama.llama_n_ctx(ctx)) break;

        if (llama.llama_decode(ctx, batch) != 0) return error.DecodeFailed;

        new_token_id = llama.llama_sampler_sample(smpl, ctx, -1);
        if (llama.llama_vocab_is_eog(vocab, new_token_id)) break;

        var buf: [4096]u8 = undefined;
        const len = llama.llama_token_to_piece(vocab, new_token_id, &buf, buf.len, 0, true);
        if (len < 0) return error.TokenToPieceFailed;

        const slice = buf[0..@as(usize, @intCast(len))];
        try response.appendSlice(allocator, slice);
        completion_tokens += 1;

        if (writer) |w| {
            w.print("{s}", .{slice}) catch |err| {
                switch (err) {
                    std.io.Writer.Error.WriteFailed => {
                        std.log.err("Error writing to output: {}\n", .{err});
                        return err;
                    },
                }
            };
            w.flush() catch |err| {
                switch (err) {
                    std.io.Writer.Error.WriteFailed => {
                        std.log.err("Error flushing to output: {}\n", .{err});
                        return err;
                    },
                }
            };
        }

        batch = llama.llama_batch_get_one(&new_token_id, 1);
    }

    return .{
        .content = try response.toOwnedSlice(allocator),
        .prompt_tokens = n_prompt,
        .completion_tokens = completion_tokens,
    };
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

pub const Tokenize = struct { vocab: *const llama.struct_llama_vocab, n_prompt: i32, prompt_tokens: []i32 };

pub fn tokenize(model: *llama_model.LlamaModel, allocator: std.mem.Allocator, prompt: []const u8) !Tokenize {
    const vocab = llama.llama_model_get_vocab(model);

    const n_prompt = -llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), null, 0, true, true);
    const prompt_tokens = try allocator.alloc(i32, @as(usize, @intCast(n_prompt)));
    // Note: no defer free — caller owns prompt_tokens via the returned Tokenize struct.
    if (llama.llama_tokenize(vocab, prompt.ptr, @as(i32, @intCast(prompt.len)), prompt_tokens.ptr, n_prompt, true, true) < 0) {
        allocator.free(prompt_tokens);
        return error.TokenizationFailed;
    }
    return Tokenize{
        .vocab = vocab.?,
        .n_prompt = n_prompt,
        .prompt_tokens = prompt_tokens,
    };
}

pub fn llama_context(model: *llama_model.LlamaModel, n_ctx: u32) !*llama.struct_llama_context {
    const cpu = std.Thread.getCpuCount() catch 4;
    // Use all physical cores for both generation and batch.
    // On Apple Silicon, hw.physicalcpu is P+E total.  For small models (≤3B)
    // E-cores can fully participate; for large models the memory-bandwidth ceiling
    // is hit before thread count matters.  Matches Ollama's behaviour (n_threads =
    // n_threads_batch = systemInfo.ThreadCount = total physical cores).
    const phys_cpu: usize = blk: {
        if (comptime @import("builtin").os.tag == .macos) {
            var val: c_uint = 0;
            var size: usize = @sizeOf(c_uint);
            if (std.c.sysctlbyname("hw.physicalcpu", &val, &size, null, 0) == 0 and val > 0) {
                break :blk @as(usize, @intCast(val));
            }
        }
        break :blk cpu;
    };

    var ctx_params = llama.llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    // n_batch must be >= the longest prompt we ever pass to llama_decode in
    // one call.  Multi-turn conversations accumulate tokens fast; setting it
    // equal to n_ctx guarantees any valid prompt fits without an assert abort.
    // llama.cpp internally splits the logical batch into n_ubatch-sized GPU
    // dispatches, so n_ubatch stays small for efficient decode.
    ctx_params.n_batch = n_ctx;
    ctx_params.n_ubatch = 512; // physical micro-batch sent to Metal per dispatch
    ctx_params.n_threads = @as(i32, @intCast(phys_cpu));
    ctx_params.n_threads_batch = @as(i32, @intCast(phys_cpu));
    const ctx = llama.llama_init_from_model(@ptrCast(model), ctx_params);
    if (ctx == null) {
        std.debug.print("Failed to create llama context", .{});
        return error.FailedToContextCreation;
    }
    return ctx.?;
}

pub fn llama_sampler() [*c]llama.struct_llama_sampler {
    // Greedy sampler only: picks argmax and short-circuits the chain.
    // min_p / temp / dist after greedy are dead code — removed.
    const smpl = llama.llama_sampler_chain_init(llama.llama_sampler_chain_default_params());
    llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_greedy());
    return smpl;
}

/// Build a sampler chain with a lazy GBNF grammar sampler followed by greedy.
///
/// The grammar is inactive until the model emits text matching `trigger`
/// (e.g. "<tool_call>").  Once the trigger fires, every subsequent token must
/// satisfy the grammar — enforcing valid tool-call JSON without constraining
/// the prefix (think / prose before the tool invocation).
///
/// `grammar_z`  — null-terminated GBNF grammar string (caller owns, may free
///                after this call returns; llama.cpp copies the string).
/// `trigger`    — null-terminated pattern string matched against accumulated
///                decoded text; grammar activates on first match.
pub fn llamaSamplerWithGrammar(
    vocab: *const llama.struct_llama_vocab,
    grammar_z: [:0]const u8,
    trigger: [*:0]const u8,
) [*c]llama.struct_llama_sampler {
    const smpl = llama.llama_sampler_chain_init(llama.llama_sampler_chain_default_params());
    // Must be var so &trigger_patterns coerces to [*c][*c]const u8 (drops const).
    var trigger_patterns = [_][*c]const u8{trigger};
    const lazy = llama.llama_sampler_init_grammar_lazy_patterns(
        vocab,
        grammar_z.ptr,
        "root",
        &trigger_patterns,
        1,
        null,
        0,
    );
    llama.llama_sampler_chain_add(smpl, lazy);
    llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_greedy());
    return smpl;
}
