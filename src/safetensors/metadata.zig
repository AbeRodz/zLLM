const std = @import("std");
const Value = @import("../ggml/KV.zig").Value;
const TensorInfo = @import("./tensor_info.zig").TensorInfo;
const TensorEntry = @import("./types.zig").TensorEntry;
const OffsetEntry = @import("./types.zig").OffsetEntry;

pub const Metadata = struct {
    model_name: []const u8,
    metadata: std.StringArrayHashMap(Value),
    tensors: std.ArrayList(TensorInfo),
    index_map: std.StringHashMap(usize),
    pub fn init(
        allocator: std.mem.Allocator,
        model_name: []const u8,
        meta_opt: ?std.StringArrayHashMap(Value),
        tensor_entries: []TensorEntry,
    ) !Metadata {
        var index_map = std.StringHashMap(usize).init(allocator);
        var tensors = std.ArrayList(TensorInfo).init(allocator);

        for (0.., tensor_entries) |i, entry| {
            try index_map.put(entry.name, i);
            try tensors.append(entry.info);
        }

        const metadata = Metadata{
            .model_name = model_name,
            .metadata = meta_opt orelse std.StringArrayHashMap(Value).init(allocator),
            .tensors = tensors,
            .index_map = index_map,
        };

        return metadata;
    }
    pub fn put(self: *Metadata, allocator: std.mem.Allocator, key_suffix: []const u8, value: Value) !void {
        const full_key_tmp = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ self.model_name, key_suffix });
        // Allocate a buffer just large enough to hold the string and copy it
        const full_key = try allocator.alloc(u8, full_key_tmp.len);
        @memcpy(full_key, full_key_tmp);
        allocator.free(full_key_tmp); // free tmp buffer after copying

        // Insert the persistent key string
        try self.metadata.put(full_key, value);
        std.debug.print("Inserting key: {s}\n", .{full_key});
    }
    pub fn get(self: *Metadata, allocator: std.mem.Allocator, key_suffix: []const u8) ?Value {
        // Build the full key just like in `put`
        const full_key_tmp = std.fmt.allocPrint(allocator, "{s}.{s}", .{ self.model_name, key_suffix }) catch return null;
        defer allocator.free(full_key_tmp);

        return self.metadata.get(full_key_tmp);
    }

    pub fn validate(self: *const Metadata) !usize {
        var start: u64 = 0;

        for (self.tensors.items, 0..) |tensor, i| {
            const s = tensor.data_offsets.start;
            const e = tensor.data_offsets.end;

            if (s != start or e < s) {
                var tensor_name = "no_tensor";
                var iter = self.index_map.iterator();
                while (iter.next()) |entry| {
                    if (entry.value_ptr.* == i) {
                        tensor_name = entry.key_ptr.*;
                        break;
                    }
                }
                std.debug.print("Invalid offset in tensor: {s}\n", .{tensor_name});
                return error.InvalidOffset;
            }

            start = e;

            var nelements: usize = 1;
            for (tensor.shape) |dim| {
                nelements = std.math.mul(usize, nelements, @as(usize, @intCast(dim))) catch {
                    return error.ValidationOverflow;
                };
            }

            const nbits = std.math.mul(usize, nelements, info.bitsize()) catch {
                return error.ValidationOverflow;
            };

            if (nbits % 8 != 0) return error.MisalignedSlice;

            const size = nbits / 8;
            const actual_size = @as(usize, @intCast(e - s));
            if (actual_size != size) return error.TensorInvalidInfo;
        }

        return @as(usize, @intCast(start));
    }
    pub fn info(self: *Metadata, name: []const u8) ?*const TensorInfo {
        const index = self.index_map.get(name) orelse return null;
        return &self.tensors.items[index.*];
    }
    pub fn getTensors(self: *Metadata, allocator: std.mem.Allocator) !std.StringHashMap(*const TensorInfo) {
        var map = std.StringHashMap(*const TensorInfo).init(allocator);

        var iter = self.index_map.iterator();
        while (iter.next()) |entry| {
            const name = entry.key_ptr.*;
            const index = entry.value_ptr.*;
            try map.put(name, &self.tensors.items[index.*]);
        }

        return map;
    }
    pub fn offsetKeys(self: Metadata, allocator: std.mem.Allocator) ![][]const u8 {
        var kvs = try allocator.alloc(OffsetEntry, self.index_map.count());

        var i: usize = 0;
        var iter = self.index_map.iterator();
        while (iter.next()) |entry| {
            kvs[i] = OffsetEntry{ .name = entry.key_ptr.*, .index = entry.value_ptr.* };
            i += 1;
        }

        std.mem.sort(
            OffsetEntry,
            kvs,
            {}, // context
            struct {
                pub fn lessThan(_: void, lhs: OffsetEntry, rhs: OffsetEntry) bool {
                    return std.mem.lessThan(u8, lhs.name, rhs.name);
                }
            }.lessThan,
        );

        var result = try allocator.alloc([]const u8, kvs.len);
        for (kvs, 0..) |kv, j| {
            result[j] = kv.name;
        }

        return result;
    }
    pub fn dataLen(self: *Metadata) usize {
        if (self.tensors.items.len == 0) return 0;
        return @as(usize, @intCast(self.tensors.items[self.tensors.items.len - 1].data_offsets.end));
    }
    pub fn getMetadata(self: *Metadata) *const std.StringHashMap([]const u8) {
        return &self.metadata;
    }
};
