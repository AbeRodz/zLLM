const std = @import("std");
const tk = @import("tokamak");
const handlers = @import("handlers.zig");
const model_handlers = @import("model_handlers.zig");
pub const routes: []const tk.Route = &.{
    .get("/health", handlers.handleHealth),
    .post0("/v3/chat/completions", handlers.handleCompletion),
    .post0("/v2/chat/completions", handlers.handleCompletions),
    .post0("/v1/chat/completions", handlers.handleCompletionStream),
    .get("/v1/models", model_handlers.handleAvailableModels),
};
