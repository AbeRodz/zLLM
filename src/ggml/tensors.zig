const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub const Layer = std.StringHashMapUnmanaged(*const Tensor);
pub const Tensors = struct {
    items: std.ArrayList(Tensor),
    pub fn init(allocator: std.mem.Allocator) Tensors {
        return Tensors{
            .items = std.ArrayList(Tensor).init(allocator),
        };
    }
    pub fn itemsWithPrefix(self: *const Tensors, allocator: std.mem.Allocator, prefix: ?[]const u8) !std.ArrayList(*const Tensor) {
        var result = std.ArrayList(*const Tensor).init(allocator);

        for (self.items.items) |*t| {
            if (prefix == null or std.mem.startsWith(u8, t.name, prefix.?)) {
                try result.append(t);
            }
        }

        return result;
    }
    pub fn groupLayers(self: Tensors, allocator: std.mem.Allocator) !std.StringHashMapUnmanaged(Layer) {
        var layers = std.StringHashMapUnmanaged(*Layer){};

        for (self.items.items) |t| {
            var parts = std.ArrayList([]const u8).init(allocator);
            defer parts.deinit();
            try splitDot(allocator, t.name, &parts);

            const index_opt = try indexFunc(parts.items, blkOrMm);

            if (index_opt) |index| {
                if (parts.items.len > index + 2) {
                    // Join parts up to index+2 as a single string
                    const joined = try std.mem.join(allocator, ".", parts.items[0 .. index + 2]);
                    defer allocator.free(joined);

                    // Build new parts array with joined prefix + suffix
                    try parts.resize(0);
                    try parts.append(joined);
                    for (parts.items[index + 2 ..]) |s| try parts.append(s);
                }
            }

            const key = parts.items[0];
            const subkey = try std.mem.join(allocator, ".", parts.items[1..]);
            defer allocator.free(subkey);

            _ = try layers.getOrPut(allocator, key);
            const layer_ptr: Layer = layers.getPtr(key).?;
            if (layer_ptr.*.capacity == 0) {
                layer_ptr.* = Layer.init(allocator);
            }

            try layer_ptr.put(allocator, subkey, &t);
        }

        return layers;
    }
};

// Helper: split string by '.'
fn splitDot(input: []const u8, parts: *std.ArrayList([]const u8)) !void {
    var it = std.mem.split(input, ".");
    while (it.next()) |part| {
        try parts.append(part);
    }
}

// Helper: find index where predicate is true
fn indexFunc(parts: []const []const u8, pred: fn ([]const u8) bool) !?usize {
    for (parts, 0..) |p, i| {
        if (pred(p)) return i;
    }
    return null;
}

// Predicate for "blk" or "mm"
fn blkOrMm(s: []const u8) bool {
    return std.mem.eql(u8, s, "blk") or std.mem.eql(u8, s, "mm");
}

pub fn size(layer: Layer) u64 {
    var total: u64 = 0;

    var it = layer.iterator();
    while (it.next()) |entry| {
        const tensor = entry.value_ptr.*;
        total += tensor.size();
    }

    return total;
}
