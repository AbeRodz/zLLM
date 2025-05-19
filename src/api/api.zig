const std = @import("std");
const tk = @import("tokamak");
const handlers = @import("handlers.zig");
pub const routes: []const tk.Route = &.{
    .post0("/v3/chat/completions", handlers.handleCompletion),
    .post0("/v2/chat/completions", handlers.handleCompletions), // debugging purposes, just expects a request and prints its body
    .post0("/v1/chat/completions", handlers.handleCompletionStream),
};
