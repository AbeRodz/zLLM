const std = @import("std");
const TensorType = @import("types.zig").TensorType;
const TensorTypeBlockSize = @import("types.zig").blockSize;
const TensorTypeSize = @import("types.zig").TypeSize;
const TensorTypeString = @import("types.zig").String;

pub const Tensor = struct {
    name: []const u8,
    shape: []u64,
    offset: u64,
    kind: u32,

    const Self = @This();

    pub fn block(self: Self) i32 {
        const prefix = "blk.";
        if (!std.mem.startsWith(u8, self.name, prefix)) return -1;

        const dot_index = std.mem.indexOfScalar(u8, self.name[prefix.len..], '.') orelse return -1;
        const num_slice = self.name[prefix.len .. prefix.len + dot_index];

        return std.fmt.parseInt(i32, num_slice, 10) catch -1;
    }
    pub fn blockSize(self: Self) u64 {
        const tensor_type = std.meta.intToEnum(TensorType, self.kind) catch return 256;
        return TensorTypeBlockSize(tensor_type);
    }

    pub fn typeSize(self: Self) u64 {
        const tensor_type = std.meta.intToEnum(TensorType, self.kind) catch return 256;
        return TensorTypeSize(tensor_type);
    }

    pub fn elements(self: Self) u64 {
        var count: u64 = 1;

        for (self.shape) |n| {
            count *= n;
        }
        return count;
    }

    pub fn size(self: Self) u64 {
        return self.elements() * self.typeSize() / self.blockSize();
    }
    pub fn tensorType(self: Self) []const u8 {
        const tensor_type = std.meta.intToEnum(TensorType, self.kind) catch return 256;
        return TensorTypeString(tensor_type);
    }
};
