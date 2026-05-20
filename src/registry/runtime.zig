const zllama = @import("../llama/llama.zig");
const registry = @import("model_registry.zig");
const converter = @import("../safetensors/gguf/convert.zig");
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

/// Parse "gemma3:q8" → { model: "gemma3", qtype: .q8_0 }.
/// No colon → qtype defaults to .f16.
fn parseModelSpec(spec: []const u8) struct { model: []const u8, qtype: converter.QuantType } {
    if (std.mem.indexOf(u8, spec, ":")) |sep| {
        const tag = spec[sep + 1 ..];
        const qtype: converter.QuantType =
            if (std.mem.eql(u8, tag, "q8") or std.mem.eql(u8, tag, "q8_0")) .q8_0
            else if (std.mem.eql(u8, tag, "q4k") or std.mem.eql(u8, tag, "q4_k")) .q4_k
            else .f16;
        return .{ .model = spec[0..sep], .qtype = qtype };
    }
    return .{ .model = spec, .qtype = .f16 };
}

/// Return the cached model for model_name, loading it from disk on first use.
/// model_name may encode a quant suffix via colon: "gemma3:q8".
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

    const spec = parseModelSpec(model_name);
    const model = try zllama.loadLlamaModelFromRegistry(spec.model, spec.qtype, allocator);
    const entry = LoadedModel{ .model = @ptrCast(model) };

    // Use page_allocator for the map key so it outlives any request arena.
    const key = try std.heap.page_allocator.dupe(u8, model_name);
    errdefer std.heap.page_allocator.free(key);
    try runtime_store.put(key, entry);

    return entry;
}
