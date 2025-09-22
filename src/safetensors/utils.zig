const std = @import("std");

pub fn bytesToF32(bytes: []const u8) f32 {
    const raw: u32 = std.mem.readInt(u32, bytes[0..4], .little);
    return @as(f32, @bitCast(raw));
}

pub fn bf16ToF32(bits: u16) f32 {
    const u32_bits: u32 = @as(u32, bits) << 16;
    return @bitCast(u32_bits);
}

pub fn bytesToBF16(bytes: []const u8) f32 {
    const raw: u16 = std.mem.readInt(u16, bytes[0..2], .little);
    return bf16ToF32(raw);
}

pub fn shape_len_product(shape: []u64) usize {
    var n: usize = 1;
    for (shape) |d| {
        n *= @as(usize, @intCast(d));
    }
    return n;
}
