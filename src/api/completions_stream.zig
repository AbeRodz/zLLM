const std = @import("std");
const StreamIter = @import("../llama/llama.zig").StreamIter;
const completions = @import("completions.zig");
const StreamOptions = @import("completions.zig").StreamOptions;
const ResponseMessage = @import("completions.zig").ResponseMessage;
const ChunkChoice = @import("completions.zig").ChunkChoice;
const Content = @import("completions.zig").Content;
const ChatCompletionRequest = @import("completions.zig").ChatCompletionRequest;
const ChatCompletionChunk = @import("completions.zig").ChatCompletionChunk;
const llama = @import("../llama/llama.zig");

pub const CompletionStreamer = struct {
    allocator: std.mem.Allocator,
    iter: StreamIter,
    model: []const u8,

    pub fn init(allocator: std.mem.Allocator, req: ChatCompletionRequest, prompt: []const u8) !CompletionStreamer {
        return CompletionStreamer{
            .allocator = allocator,
            .iter = try llama.respondToPromptStream(allocator, req.model, 8192, prompt),
            .model = req.model,
        };
    }

    pub fn next(self: *CompletionStreamer) !?ChatCompletionChunk {
        const maybeChunk = try self.iter.next();

        if (maybeChunk) |chunk| {
            const choices = try std.heap.page_allocator.alloc(ChunkChoice, 1);
            const stdout = std.io.getStdOut().writer();
            //try stdout.print("{s}", .{chunk});

            choices[0] = ChunkChoice{
                .delta = ResponseMessage{
                    .content = chunk,
                    .role = "assistant",
                },
                .index = 0,
                .finish_reason = null,
            };
            try stdout.print("{s}", .{choices[0].delta.content});
            return completions.ChatCompletionChunk{
                .id = "chatcmpl-stream",
                .choices = choices,
                .model = self.model,
                .created = std.time.timestamp(),
                .system_fingerprint = "zLLM",
                .object = "chat.completion.chunk",
                .usage = .{
                    .prompt_tokens = self.iter.prompt_token_count,
                    .completion_tokens = self.iter.completion_token_count,
                    .total_tokens = self.iter.prompt_token_count + self.iter.completion_token_count,
                },
            };
        } else {
            return null;
        }
    }

    pub fn deinit(self: *CompletionStreamer) void {
        self.iter.deinit();
    }
};
