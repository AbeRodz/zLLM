const zllama = @import("../llama/llama.zig");
const registry = @import("model_registry.zig");
const std = @import("std");

pub const llama = @cImport({
    @cInclude("llama.h");
});

const llama_model = @import("../llama/cTypes.zig").LlamaModel;

/// A loaded model whose weights are cached globally for the lifetime of the
/// process.  The llama_context (KV cache, batch buffers) is intentionally NOT
/// stored here — each session owns its own context via SessionManager.
pub const LoadedModel = struct {
    model: *llama.struct_llama_model,
};

var runtime_store = std.StringHashMap(LoadedModel).init(std.heap.page_allocator);
var lock = std.Thread.Mutex{};

/// Return the cached model for model_name, loading it from disk on first use.
/// Model weights are shared read-only across all requests and sessions.
pub fn getOrLoadModel(
    allocator: std.mem.Allocator,
    model_name: []const u8,
) !LoadedModel {
    lock.lock();
    defer lock.unlock();

    if (runtime_store.get(model_name)) |cached| {
        return cached;
    }

    const model = try zllama.loadLlamaModelFromRegistry(model_name, allocator);
    const entry = LoadedModel{ .model = @ptrCast(model) };

    // Use page_allocator for the map key so it outlives any request arena.
    const key = try std.heap.page_allocator.dupe(u8, model_name);
    errdefer std.heap.page_allocator.free(key);
    try runtime_store.put(key, entry);

    return entry;
}
