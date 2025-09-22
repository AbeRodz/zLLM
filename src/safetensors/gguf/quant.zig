const std = @import("std");

// convert IEEE-754 binary16 (half) to f32
pub fn halfToF32(bits16: u16) f32 {
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
// convert f32 to IEEE-754 binary16 (half)
pub fn f32ToHalf(val: f32) u16 {
    const f32_bits: u32 = @as(u32, @bitCast(val));

    const sign: u16 = @as(u16, @intCast((f32_bits >> 16) & 0x8000));

    var exponent: i32 = @as(i32, @intCast((f32_bits >> 23) & 0xFF)) - 127 + 15;

    //var exponent: u16 = @as(u16, @intCast((f32_bits >> 23) & 0xFF)) - 127 + 15;
    var mantissa: u16 = @as(u16, @intCast((f32_bits >> 13) & 0x3FF));

    // handle overflow, underflow, NaN, and zero
    if (exponent <= 0) {
        // subnormal or zero
        if (exponent < -10) return sign; // too small -> zero
        mantissa = @as(u16, (mantissa | 0x400) >> @as(u4, @intCast(1 - exponent)));

        exponent = 0;
    } else if (exponent >= 0x1F) {
        // inf or NaN
        exponent = 0x1F;
        mantissa = if ((f32_bits & 0x007FFFFF) != 0) 0x200 else 0;
    }

    return sign | @as(u16, @intCast(exponent << 10)) | mantissa;
}
