const std = @import("std");

pub const GGUFDataType = enum(u32) {
    ggufTypeUint8 = 0,
    ggufTypeInt8 = 1,
    ggufTypeUint16 = 2,
    ggufTypeInt16 = 3,
    ggufTypeUint32 = 4,
    ggufTypeInt32 = 5,
    ggufTypeFloat32 = 6,
    ggufTypeBool = 7,
    ggufTypeString = 8,
    ggufTypeArray = 9,
    ggufTypeUint64 = 10,
    ggufTypeInt64 = 11,
    ggufTypeFloat64 = 12,
};

pub const TensorType = enum(u32) {
    TensorTypeF32 = 0,
    TensorTypeF16,
    TensorTypeQ4_0,
    TensorTypeQ4_1,
    tensorTypeQ4_2, // unused by GGML
    tensorTypeQ4_3, // unused by GGML
    TensorTypeQ5_0,
    TensorTypeQ5_1,
    TensorTypeQ8_0,
    TensorTypeQ8_1,
    TensorTypeQ2_K,
    TensorTypeQ3_K,
    TensorTypeQ4_K,
    TensorTypeQ5_K,
    TensorTypeQ6_K,
    TensorTypeQ8_K,
    tensorTypeIQ2_XXS,
    tensorTypeIQ2_XS,
    tensorTypeIQ3_XXS,
    tensorTypeIQ1_S,
    tensorTypeIQ4_NL,
    tensorTypeIQ3_S,
    tensorTypeIQ2_S,
    tensorTypeIQ4_XS,
    TensorTypeI8,
    TensorTypeI16,
    TensorTypeI32,
    TensorTypeI64,
    TensorTypeF64,
    tensorTypeIQ1_M,
    TensorTypeBF16,
    tensorTypeQ4_0_4_4, // unused by GGML
    tensorTypeQ4_0_4_8, // unused by GGML
    tensorTypeQ4_0_8_8, // unused by GGML
    tensorTypeTQ1_0,
    tensorTypeTQ2_0,
    tensorTypeIQ4_NL_4_4, // unused by GGML
    tensorTypeIQ4_NL_4_8, // unused by GGML
    tensorTypeIQ4_NL_8_8, // unused by GGML
};

pub fn ParseTensorType(s: []const u8) !TensorType {
    return switch (std.mem.eql(u8, s, "F32")) {
        true => TensorType.TensorTypeF32,
        false => switch (std.mem.eql(u8, s, "F16")) {
            true => TensorType.TensorTypeF16,
            false => switch (std.mem.eql(u8, s, "Q4_0")) {
                true => TensorType.TensorTypeQ4_0,
                false => switch (std.mem.eql(u8, s, "Q4_1")) {
                    true => TensorType.TensorTypeQ4_1,
                    false => switch (std.mem.eql(u8, s, "Q5_0")) {
                        true => TensorType.TensorTypeQ5_0,
                        false => switch (std.mem.eql(u8, s, "Q5_1")) {
                            true => TensorType.TensorTypeQ5_1,
                            false => switch (std.mem.eql(u8, s, "Q8_0")) {
                                true => TensorType.TensorTypeQ8_0,
                                false => switch (std.mem.eql(u8, s, "Q8_1")) {
                                    true => TensorType.TensorTypeQ8_1,
                                    false => switch (std.mem.eql(u8, s, "Q2_K")) {
                                        true => TensorType.TensorTypeQ2_K,
                                        false => switch (std.mem.eql(u8, s, "Q3_K")) {
                                            true => TensorType.TensorTypeQ3_K,
                                            false => switch (std.mem.eql(u8, s, "Q4_K")) {
                                                true => TensorType.TensorTypeQ4_K,
                                                false => switch (std.mem.eql(u8, s, "Q5_K")) {
                                                    true => TensorType.TensorTypeQ5_K,
                                                    false => switch (std.mem.eql(u8, s, "Q6_K")) {
                                                        true => TensorType.TensorTypeQ6_K,
                                                        false => switch (std.mem.eql(u8, s, "Q8_K")) {
                                                            true => TensorType.TensorTypeQ8_K,
                                                            false => switch (std.mem.eql(u8, s, "F64")) {
                                                                true => TensorType.TensorTypeF64,
                                                                false => switch (std.mem.eql(u8, s, "BF16")) {
                                                                    true => TensorType.TensorTypeBF16,
                                                                    false => return error.UnsupportedQuantizationType,
                                                                },
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    };
}

pub fn isQuantized(t: TensorType) bool {
    return switch (t) {
        .TensorTypeF32, .TensorTypeF16, .TensorTypeBF16 => false,
        else => true,
    };
}

pub fn rowSize(t: TensorType, ne: u64) u64 {
    return (TypeSize(t) * ne) / blockSize(t);
}

pub fn TypeSize(t: TensorType) u64 {
    const block_size = blockSize(t);

    return switch (t) {
        .TensorTypeF32 => 4,
        .TensorTypeF16 => 2,
        .TensorTypeQ4_0 => 2 + block_size / 2,
        .TensorTypeQ4_1 => 2 + 2 + block_size / 2,
        .TensorTypeQ5_0 => 2 + 4 + block_size / 2,
        .TensorTypeQ5_1 => 2 + 2 + 4 + block_size / 2,
        .TensorTypeQ8_0 => 2 + block_size,
        .TensorTypeQ8_1 => 2 + 2 + block_size,
        .TensorTypeQ2_K => block_size / 16 + block_size / 4 + 2 + 2,
        .TensorTypeQ3_K => block_size / 8 + block_size / 4 + 12 + 2,
        .TensorTypeQ4_K => 2 + 2 + 12 + block_size / 2,
        .TensorTypeQ5_K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
        .TensorTypeQ6_K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
        .TensorTypeQ8_K => 4 + block_size + 2 * block_size / 16,
        .tensorTypeIQ2_XXS => 2 + 2 * block_size / 8,
        .tensorTypeIQ2_XS => 2 + 2 * block_size / 8 + block_size / 32,
        .tensorTypeIQ3_XXS => 2 + block_size / 4 + block_size / 8,
        .tensorTypeIQ1_S => 2 + block_size / 8 + block_size / 16,
        .tensorTypeIQ4_NL => 2 + block_size / 2,
        .tensorTypeIQ3_S => 2 + block_size / 4 + block_size / 8 + block_size / 32 + 4,
        .tensorTypeIQ2_S => 2 + block_size / 4 + block_size / 16,
        .tensorTypeIQ4_XS => 2 + 2 + block_size / 2 + block_size / 64,
        .TensorTypeI8 => 1,
        .TensorTypeI16 => 2,
        .TensorTypeI32 => 4,
        .TensorTypeI64 => 8,
        .TensorTypeF64 => 8,
        .tensorTypeIQ1_M => block_size / 8 + block_size / 16 + block_size / 32,
        .TensorTypeBF16 => 2,
        else => 0,
    };
}

pub fn blockSize(t: TensorType) u64 {
    return switch (t) {
        .TensorTypeF32, .TensorTypeF16, .TensorTypeI8, .TensorTypeI16, .TensorTypeI32, .TensorTypeI64, .TensorTypeF64, .TensorTypeBF16 => 1,

        .TensorTypeQ4_0, .TensorTypeQ4_1, .TensorTypeQ5_0, .TensorTypeQ5_1, .TensorTypeQ8_0, .TensorTypeQ8_1, .tensorTypeIQ4_NL => 32,

        else => 256,
    };
}

pub fn String(t: TensorType) []const u8 {
    return switch (t) {
        .TensorTypeF32 => "F32",
        .TensorTypeF16 => "F16",
        .TensorTypeQ4_0 => "Q4_0",
        .TensorTypeQ4_1 => "Q4_1",
        .TensorTypeQ5_0 => "Q5_0",
        .TensorTypeQ5_1 => "Q5_1",
        .TensorTypeQ8_0 => "Q8_0",
        .TensorTypeQ8_1 => "Q8_1",
        .TensorTypeQ2_K => "Q2_K",
        .TensorTypeQ3_K => "Q3_K",
        .TensorTypeQ4_K => "Q4_K",
        .TensorTypeQ5_K => "Q5_K",
        .TensorTypeQ6_K => "Q6_K",
        .TensorTypeQ8_K => "Q8_K",
        .TensorTypeF64 => "F64",
        .TensorTypeBF16 => "BF16",
        else => "unknown",
    };
}
