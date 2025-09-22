const std = @import("std");
const tk = @import("tokamak");
const models = @import("models/available_models.zig");

pub fn handleAvailableModels(allocator: std.mem.Allocator) !models.AvailableModelsResponse {
    const available_models = try models.AvailableModels(allocator);

    return available_models;
}
