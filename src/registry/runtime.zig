const zllama = @import("../llama/llama.zig");
const registry = @import("model_registry.zig");
const std = @import("std");
pub const llama = @cImport({
    @cInclude("llama.h");
});

pub const LoadedModel = struct {
    model: *llama.struct_llama_model,
    ctx: *llama.struct_llama_context,
};

var runtime_store = std.StringHashMap(LoadedModel).init(std.heap.page_allocator);
var lock = std.Thread.Mutex{};

// When first request reaches it loads from .cache (cold start) else from the runtime cache (hot start).
pub fn getOrLoadModel(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    n_ctx: u32,
) !LoadedModel {
    lock.lock();
    defer lock.unlock();

    if (runtime_store.get(model_name)) |cached| {
        return cached;
    }
    const model = try zllama.loadLlamaModelFromRegistry(model_name);
    const ctx = try zllama.llama_context(model, n_ctx);
    const entry = LoadedModel{ .model = model, .ctx = ctx };

    try runtime_store.put(try allocator.dupe(u8, model_name), entry);
    return entry;
}
