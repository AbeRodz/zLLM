const std = @import("std");
const registry = @import("../../registry/model_registry.zig");

/// Matches the OpenAI  GET /v1/models  individual model object.
pub const ModelObject = struct {
    id: []const u8,
    object: []const u8 = "model",
    created: i64,
    owned_by: []const u8 = "local",
};

/// Matches the OpenAI  GET /v1/models  response envelope.
pub const AvailableModelsResponse = struct {
    object: []const u8 = "list",
    data: []ModelObject,
};

pub fn AvailableModels(allocator: std.mem.Allocator) !AvailableModelsResponse {
    const models = registry.listAvailableModels(allocator) catch |err| {
        std.debug.print("Error listing models: {}\n", .{err});
        return err;
    };
    defer allocator.free(models);

    var model_list = try std.ArrayListUnmanaged(ModelObject).initCapacity(allocator, models.len);
    const now = std.time.timestamp();

    for (models) |model| {
        try model_list.append(allocator, ModelObject{
            .id = model.name,
            .created = now,
        });
    }

    return AvailableModelsResponse{
        .data = model_list.items,
    };
}
