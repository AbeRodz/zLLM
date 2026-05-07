const std = @import("std");
const uuid = @import("uuid");
const StreamIter = @import("../llama/llama.zig").StreamIter;
const completions = @import("models/completions.zig");
const StreamOptions = @import("models/completions.zig").StreamOptions;
const StreamDelta = @import("models/completions.zig").StreamDelta;
const ChunkChoice = @import("models/completions.zig").ChunkChoice;
const ChatCompletionRequest = @import("models/completions.zig").ChatCompletionRequest;
const ChatCompletionChunk = @import("models/completions.zig").ChatCompletionChunk;
const llama = @import("../llama/llama.zig");
const chat_format = llama.chat_format;

pub const CompletionStreamer = struct {
    allocator: std.mem.Allocator,
    iter: StreamIter,
    model: []const u8,
    id: []const u8,
    /// Unix timestamp captured at request start; reused for every chunk so
    /// we avoid a syscall per generated token.
    created: i64,
    /// True until the first content chunk has been sent (role is only in first delta).
    first_chunk: bool,
    /// True after the stop chunk has been returned; subsequent next() calls return null.
    finished: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        req: ChatCompletionRequest,
        messages: []const llama.ApiMessage,
        /// Optional session ID injected by the gateway via X-Session-ID header.
        /// Null means stateless: a fresh context is created and destroyed per request.
        session_id: ?[]const u8,
        /// Tool definitions — passed through to respondToPromptStream so the
        /// lazy GBNF grammar sampler is installed when tools are present.
        tools: ?[]const chat_format.ApiTool,
    ) !CompletionStreamer {
        // Dupe model name and id into allocator so they survive after the handler returns.
        const model_copy = try allocator.dupe(u8, req.model);

        const urn = uuid.urn.serialize(uuid.v4.new());
        const id_prefix = "chatcmpl-";
        const id = try allocator.alloc(u8, id_prefix.len + urn.len);
        @memcpy(id[0..id_prefix.len], id_prefix);
        @memcpy(id[id_prefix.len..], &urn);

        return CompletionStreamer{
            .allocator = allocator,
            .iter = try llama.respondToPromptStream(allocator, req.model, 8192, messages, session_id, tools),
            .model = model_copy,
            .id = id,
            .created = std.time.timestamp(),
            .first_chunk = true,
            .finished = false,
        };
    }

    pub inline fn next(self: *CompletionStreamer) !?ChatCompletionChunk {
        if (self.finished) return null;

        // EndOfStream is the normal end-of-generation signal from StreamIter —
        // treat it as null (no more tokens) rather than propagating as an error.
        const maybeChunk = nosuspend self.iter.next() catch |err| switch (err) {
            error.EndOfStream => null,
            else => return err,
        };

        if (maybeChunk) |chunk| {
            // OpenAI: role is only present in the very first delta.
            const delta = StreamDelta{
                .role = if (self.first_chunk) "assistant" else null,
                .content = chunk,
            };
            self.first_chunk = false;

            const choice = ChunkChoice{
                .delta = delta,
                .index = 0,
                .finish_reason = null,
            };
            return ChatCompletionChunk{
                .id = self.id,
                .choices = [1]ChunkChoice{choice},
                .model = self.model,
                .created = self.created,
                .system_fingerprint = "zLLM",
                .usage = null,
            };
        } else {
            // Iterator exhausted — emit the final stop chunk and mark done.
            self.finished = true;
            const stop_choice = ChunkChoice{
                .delta = StreamDelta{},
                .index = 0,
                .finish_reason = "stop",
            };
            return ChatCompletionChunk{
                .id = self.id,
                .choices = [1]ChunkChoice{stop_choice},
                .model = self.model,
                .created = self.created,
                .system_fingerprint = "zLLM",
                .usage = .{
                    .prompt_tokens = self.iter.prompt_token_count,
                    .completion_tokens = self.iter.completion_token_count,
                    .total_tokens = self.iter.prompt_token_count + self.iter.completion_token_count,
                },
            };
        }
    }

    /// Signal the decode loop to stop at the next token boundary.
    /// Called when the SSE write fails (client disconnected) so we don't
    /// burn GPU cycles generating tokens nobody will receive.
    pub fn cancel(self: *CompletionStreamer) void {
        self.iter.is_done = true;
    }

    pub fn deinit(self: *CompletionStreamer) void {
        // StreamIter.deinit() handles: freeing the sampler, optionally freeing
        // the ctx (stateless path), and releasing the session (session path).
        self.iter.deinit();
    }
};
