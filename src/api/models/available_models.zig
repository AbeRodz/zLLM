const std = @import("std");
const registry = @import("../../registry/model_registry.zig");

pub fn AvailableModels(allocator: std.mem.Allocator) !AvailableModelsResponse {
    const models = registry.listAvailableModels(allocator) catch |err| {
        std.debug.print("Error listing models: {}\n", .{err});
        return err;
    };
    defer allocator.free(models);

    var available_models = try std.ArrayListUnmanaged([]const u8).initCapacity(
        allocator,
        models.len,
    );

    for (models) |model| {
        try available_models.append(allocator, model.name);
    }

    return AvailableModelsResponse{
        .available_models = available_models.items,
    };
}

pub const AvailableModelsResponse = struct {
    available_models: [][]const u8,
};
