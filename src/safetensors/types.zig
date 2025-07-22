const std = @import("std");
const TensorType = @import("../ggml/types.zig").TensorType;

pub fn mapDtypeToGGML(dtype: []const u8) !TensorType {
    if (std.mem.eql(u8, dtype, "F32")) return TensorType.TensorTypeF32;
    if (std.mem.eql(u8, dtype, "F16")) return TensorType.TensorTypeF16;
    if (std.mem.eql(u8, dtype, "Q4_0")) return TensorType.TensorTypeQ4_0;
    if (std.mem.eql(u8, dtype, "Q4_1")) return TensorType.TensorTypeQ4_1;
    if (std.mem.eql(u8, dtype, "Q5_0")) return TensorType.TensorTypeQ5_0;
    if (std.mem.eql(u8, dtype, "Q5_1")) return TensorType.TensorTypeQ5_1;
    if (std.mem.eql(u8, dtype, "Q8_0")) return TensorType.TensorTypeQ8_0;
    if (std.mem.eql(u8, dtype, "Q8_1")) return TensorType.TensorTypeQ8_1;
    if (std.mem.eql(u8, dtype, "Q2_K")) return TensorType.TensorTypeQ2_K;
    if (std.mem.eql(u8, dtype, "Q3_K")) return TensorType.TensorTypeQ3_K;
    if (std.mem.eql(u8, dtype, "Q4_K")) return TensorType.TensorTypeQ4_K;
    if (std.mem.eql(u8, dtype, "Q5_K")) return TensorType.TensorTypeQ5_K;
    if (std.mem.eql(u8, dtype, "Q6_K")) return TensorType.TensorTypeQ6_K;
    if (std.mem.eql(u8, dtype, "Q8_K")) return TensorType.TensorTypeQ8_K;
    if (std.mem.eql(u8, dtype, "I8")) return TensorType.TensorTypeI8;
    if (std.mem.eql(u8, dtype, "I16")) return TensorType.TensorTypeI16;
    if (std.mem.eql(u8, dtype, "I32")) return TensorType.TensorTypeI32;
    if (std.mem.eql(u8, dtype, "I64")) return TensorType.TensorTypeI64;
    if (std.mem.eql(u8, dtype, "F64")) return TensorType.TensorTypeF64;
    if (std.mem.eql(u8, dtype, "BF16")) return TensorType.TensorTypeBF16;
    return error.UnsupportedQuantizationType;
}
