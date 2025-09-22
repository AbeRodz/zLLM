const std = @import("std");

pub const TensorInfo = struct {
    dtype: []const u8,
    shape: []u64,
    data_offsets: struct {
        start: u64,
        end: u64,
    },
    pub fn bitsize(self: *const TensorInfo) usize {
        if (std.mem.eql(u8, self.dtype, "F4")) return 4;
        if (std.mem.eql(u8, self.dtype, "F6_E3M2")) return 6;
        if (std.mem.eql(u8, self.dtype, "F6_E2M3")) return 6;
        if (std.mem.eql(u8, self.dtype, "BOOL")) return 8;
        if (std.mem.eql(u8, self.dtype, "U8")) return 8;
        if (std.mem.eql(u8, self.dtype, "I8")) return 8;
        if (std.mem.eql(u8, self.dtype, "F8_E5M2")) return 8;
        if (std.mem.eql(u8, self.dtype, "F8_E4M3")) return 8;
        if (std.mem.eql(u8, self.dtype, "F8_E8M0")) return 8;
        if (std.mem.eql(u8, self.dtype, "I16")) return 16;
        if (std.mem.eql(u8, self.dtype, "U16")) return 16;
        if (std.mem.eql(u8, self.dtype, "F16")) return 16;
        if (std.mem.eql(u8, self.dtype, "BF16")) return 16;
        if (std.mem.eql(u8, self.dtype, "I32")) return 32;
        if (std.mem.eql(u8, self.dtype, "U32")) return 32;
        if (std.mem.eql(u8, self.dtype, "F32")) return 32;
        if (std.mem.eql(u8, self.dtype, "I64")) return 64;
        if (std.mem.eql(u8, self.dtype, "U64")) return 64;
        if (std.mem.eql(u8, self.dtype, "F64")) return 64;

        return 0; // Unknown dtype
    }
};
