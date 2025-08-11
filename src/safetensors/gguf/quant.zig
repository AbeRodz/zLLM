const std = @import("std");

// convert IEEE-754 binary16 (half) to f32
fn halfToF32(bits16: u16) f32 {
    const s: u32 = @as(u32, (bits16 >> 15) & 0x1);
    const e: u32 = @as(u32, (bits16 >> 10) & 0x1F);
    const m: u32 = @as(u32, bits16 & 0x3FF);

    if (e == 0) {
        if (m == 0) {
            return @as(f32, @bitCast((s << 31)));
        }
        // subnormal
        var mant = m;
        var exp: i32 = -14;
        while ((mant & 0x400) == 0) : (mant <<= 1) {
            exp -= 1;
        }
        mant &= 0x3FF;
        const f32exp = @as(u32, @intCast((@as(i32, exp) + 127) & 0xFF));
        const f32mant = mant << 13;
        const bits = (s << 31) | (f32exp << 23) | f32mant;
        return @as(f32, @bitCast(bits));
    } else if (e == 0x1F) {
        // Inf or NaN
        const bits = (s << 31) | (0xFF << 23) | (m << 13);
        return @as(f32, @bitCast(bits));
    } else {
        // normalized
        const f32exp = @as(u32, (e - 15 + 127));
        const f32mant = m << 13;
        const bits = (s << 31) | (f32exp << 23) | f32mant;
        return @as(f32, @bitCast(bits));
    }
}
