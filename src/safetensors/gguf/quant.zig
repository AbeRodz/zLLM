const std = @import("std");

/// Quantize f32 values into Q8_0 blocks.
/// values.len must be a multiple of 32.
/// Returns caller-owned []u8 of length (values.len / 32) * 34.
pub fn quantizeTensorQ8_0(allocator: std.mem.Allocator, values: []const f32) ![]u8 {
    const block_size: usize = 32;
    std.debug.assert(values.len % block_size == 0);

    const num_blocks = values.len / block_size;
    const out = try allocator.alloc(u8, num_blocks * 34);

    for (0..num_blocks) |b| {
        const block = values[b * block_size .. (b + 1) * block_size];
        const out_block = out[b * 34 ..][0..34];

        var amax: f32 = 0.0;
        for (block) |v| {
            const av = @abs(v);
            if (av > amax) amax = av;
        }

        const d: f32 = amax / 127.0;
        const d_bits: u16 = f32ToHalf(d);
        out_block[0] = @truncate(d_bits);
        out_block[1] = @truncate(d_bits >> 8);

        const inv_d: f32 = if (d != 0.0) 1.0 / d else 0.0;
        for (block, 0..) |v, i| {
            const q_f: f32 = @round(v * inv_d);
            const q_i32: i32 = @intFromFloat(q_f);
            const q_clamped: i8 = @intCast(std.math.clamp(q_i32, -127, 127));
            out_block[2 + i] = @bitCast(q_clamped);
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Q4_K helpers (translated from ggml-quants.c)
// ---------------------------------------------------------------------------

/// Weighted least-squares solver for a 32-element sub-block.
/// Finds the best (scale, min) pair to represent values in [0, 15].
/// Direct port of make_qkx2_quants(..., nmax=15, rmin=-1, rdelta=0.1, nstep=20, use_mad=false).
fn makeQkx2Quants(
    values: []const f32,
    weights: []const f32,
    L: []u8,
    the_min: *f32,
    Laux: []u8,
) f32 {
    var vmin: f32 = values[0];
    var vmax: f32 = values[0];
    var sum_w: f32 = weights[0];
    var sum_x: f32 = weights[0] * values[0];

    for (1..values.len) |i| {
        if (values[i] < vmin) vmin = values[i];
        if (values[i] > vmax) vmax = values[i];
        sum_w += weights[i];
        sum_x += weights[i] * values[i];
    }

    if (vmin > 0) vmin = 0;
    if (vmax == vmin) {
        for (0..values.len) |i| L[i] = 0;
        the_min.* = -vmin;
        return 0.0;
    }

    var iscale: f32 = 15.0 / (vmax - vmin);
    var scale: f32 = 1.0 / iscale;
    var best_mad: f32 = 0;

    for (0..values.len) |i| {
        const li: i32 = @intFromFloat(@round(iscale * (values[i] - vmin)));
        L[i] = @intCast(std.math.clamp(li, 0, 15));
        const lf: f32 = @floatFromInt(L[i]);
        const diff = scale * lf + vmin - values[i];
        best_mad += weights[i] * diff * diff;
    }

    // Line search: nstep=20 steps from iscale*(rmin=-1 .. rdelta*20=2) + 15
    for (0..21) |is| {
        const is_f: f32 = @floatFromInt(is);
        iscale = (-1.0 + 0.1 * is_f + 15.0) / (vmax - vmin);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;

        for (0..values.len) |i| {
            const li: i32 = @intFromFloat(@round(iscale * (values[i] - vmin)));
            Laux[i] = @intCast(std.math.clamp(li, 0, 15));
            const w = weights[i];
            const lf: f32 = @floatFromInt(Laux[i]);
            sum_l  += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * values[i];
        }

        const D: f32 = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            var this_scale: f32 = (sum_w * sum_xl - sum_x * sum_l) / D;
            var this_min:   f32 = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                if (sum_l2 > 0) this_scale = sum_xl / sum_l2;
            }
            var mad: f32 = 0;
            for (0..values.len) |i| {
                const lf: f32 = @floatFromInt(Laux[i]);
                const diff = this_scale * lf + this_min - values[i];
                mad += weights[i] * diff * diff;
            }
            if (mad < best_mad) {
                @memcpy(L[0..values.len], Laux[0..values.len]);
                best_mad = mad;
                scale = this_scale;
                vmin = this_min;
            }
        }
    }

    the_min.* = -vmin;
    return scale;
}

/// Decode the j-th (scale, min) pair from the 12-byte K-quant scales array.
/// Direct port of get_scale_min_k4 from ggml-quants.c.
fn getScaleMinK4(j: usize, q: []const u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        d.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << @as(u3, 4));
        m.* = (q[j + 4] >> @as(u3, 4)) | ((q[j] >> 6) << @as(u3, 4));
    }
}

/// Quantize f32 values into Q4_K super-blocks.
/// values.len must be a multiple of 256 (QK_K).
/// Returns caller-owned []u8 of length (values.len / 256) * 144.
///
/// Block layout (144 bytes per 256 elements):
///   [0..1]   d    – f16 super-block scale for quantized sub-block scales
///   [2..3]   dmin – f16 super-block scale for quantized sub-block mins
///   [4..15]  scales[12] – 8 sub-block scales + 8 sub-block mins, 6 bits each
///   [16..143] qs[128]   – nibble-packed 4-bit quantized values
pub fn quantizeTensorQ4K(allocator: std.mem.Allocator, values: []const f32) ![]u8 {
    const QK_K: usize = 256;
    const block_bytes: usize = 144; // 2+2+12+128
    std.debug.assert(values.len % QK_K == 0);

    const num_blocks = values.len / QK_K;
    const out = try allocator.alloc(u8, num_blocks * block_bytes);
    @memset(out, 0);

    var L: [256]u8 = undefined;
    var Laux: [32]u8 = undefined;
    var weights: [32]f32 = undefined;
    var scales: [8]f32 = undefined;
    var mins: [8]f32 = undefined;

    for (0..num_blocks) |b| {
        const x = values[b * QK_K .. (b + 1) * QK_K];
        const blk = out[b * block_bytes ..][0..block_bytes];
        const sc_bytes = blk[4..16]; // 12-byte scales region

        // ── Step 1: per-sub-block weighted solver ────────────────────────────
        var max_scale: f32 = 0;
        var max_min:   f32 = 0;

        for (0..8) |j| {
            const sub = x[j * 32 ..][0..32];
            var sum_x2: f32 = 0;
            for (sub) |v| sum_x2 += v * v;
            const av_x: f32 = @sqrt(sum_x2 / 32.0);
            for (0..32) |l| weights[l] = av_x + @abs(sub[l]);

            scales[j] = makeQkx2Quants(sub, &weights, L[j * 32 ..][0..32], &mins[j], &Laux);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j]   > max_min)   max_min   = mins[j];
        }

        // ── Step 2: quantize sub-block scales/mins to 6 bits ─────────────────
        const inv_scale: f32 = if (max_scale > 0) 63.0 / max_scale else 0;
        const inv_min:   f32 = if (max_min   > 0) 63.0 / max_min   else 0;

        for (0..8) |j| {
            const ls: u8 = @intCast(std.math.clamp(
                @as(i32, @intFromFloat(@round(inv_scale * scales[j]))), 0, 63));
            const lm: u8 = @intCast(std.math.clamp(
                @as(i32, @intFromFloat(@round(inv_min * mins[j]))), 0, 63));

            if (j < 4) {
                sc_bytes[j]     = ls;
                sc_bytes[j + 4] = lm;
            } else {
                sc_bytes[j + 4]  = (ls & 0xF) | ((lm & 0xF) << @as(u3, 4));
                sc_bytes[j - 4] |= (ls >> @as(u3, 4)) << @as(u3, 6);
                sc_bytes[j]     |= (lm >> @as(u3, 4)) << @as(u3, 6);
            }
        }

        // Store super-block f16 scales
        const d_bits    = f32ToHalf(max_scale / 63.0);
        const dmin_bits = f32ToHalf(max_min   / 63.0);
        blk[0] = @truncate(d_bits);
        blk[1] = @truncate(d_bits >> 8);
        blk[2] = @truncate(dmin_bits);
        blk[3] = @truncate(dmin_bits >> 8);

        // ── Step 3: re-quantize each element using decoded scales ─────────────
        const d_f32    = halfToF32(d_bits);
        const dmin_f32 = halfToF32(dmin_bits);

        for (0..8) |j| {
            var sc: u8 = undefined;
            var m:  u8 = undefined;
            getScaleMinK4(j, sc_bytes, &sc, &m);
            const d_eff: f32 = d_f32 * @as(f32, @floatFromInt(sc));
            if (d_eff == 0) continue;
            const dm: f32 = dmin_f32 * @as(f32, @floatFromInt(m));
            const sub = x[j * 32 ..][0..32];
            for (0..32) |ii| {
                const li: i32 = @intFromFloat(@round((sub[ii] + dm) / d_eff));
                L[j * 32 + ii] = @intCast(std.math.clamp(li, 0, 15));
            }
        }

        // ── Step 4: pack nibbles into qs ──────────────────────────────────────
        // Each pair of consecutive 32-element sub-blocks shares 32 output bytes:
        //   qs[l] = L[j+l] (low nibble) | L[j+l+32] (high nibble)
        const qs = blk[16..144];
        var q_pos: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            for (0..32) |l| {
                qs[q_pos + l] = L[j + l] | (L[j + l + 32] << @as(u3, 4));
            }
            q_pos += 32;
        }
    }

    return out;
}

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
        const f32exp: u32 = e + 112; // e - 15 + 127, avoids u32 underflow when e < 15
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
