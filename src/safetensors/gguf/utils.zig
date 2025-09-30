const std = @import("std");
const registry = @import("../../registry/model_registry.zig");
const GGUFWriter = @import("../../ggml/writer.zig").GGUFWriter;
const Value = @import("../../ggml/KV.zig").Value;
const GGUFDataType = @import("../../ggml/types.zig").GGUFDataType;

pub fn writeKeyValue(writer: *GGUFWriter, key: []const u8, value: Value) !void {
    try writer.writeString(key);

    var type_id: GGUFDataType = undefined;
    var is_array = false;

    switch (value) {
        .u8 => type_id = GGUFDataType.ggufTypeUint8,
        .i8 => type_id = GGUFDataType.ggufTypeInt8,
        .u16 => type_id = GGUFDataType.ggufTypeUint16,
        .i16 => type_id = GGUFDataType.ggufTypeInt16,
        .u32 => type_id = GGUFDataType.ggufTypeUint32,
        .i32 => type_id = GGUFDataType.ggufTypeInt32,
        .u64 => type_id = GGUFDataType.ggufTypeUint64,
        .i64 => type_id = GGUFDataType.ggufTypeInt64,
        .f32 => type_id = GGUFDataType.ggufTypeFloat32,
        .f64 => type_id = GGUFDataType.ggufTypeFloat64,
        .bool => type_id = GGUFDataType.ggufTypeBool,
        .str => type_id = GGUFDataType.ggufTypeString,
        .u8s => {
            type_id = GGUFDataType.ggufTypeUint8;
            is_array = true;
        },
        .i8s => {
            type_id = GGUFDataType.ggufTypeInt8;
            is_array = true;
        },
        .u16s => {
            type_id = GGUFDataType.ggufTypeUint16;
            is_array = true;
        },
        .i16s => {
            type_id = GGUFDataType.ggufTypeInt16;
            is_array = true;
        },
        .u32s => {
            type_id = GGUFDataType.ggufTypeUint32;
            is_array = true;
        },
        .i32s => {
            type_id = GGUFDataType.ggufTypeInt32;
            is_array = true;
        },
        .u64s => {
            type_id = GGUFDataType.ggufTypeUint64;
            is_array = true;
        },
        .i64s => {
            type_id = GGUFDataType.ggufTypeInt64;
            is_array = true;
        },
        .f32s => {
            type_id = GGUFDataType.ggufTypeFloat32;
            is_array = true;
        },
        .f64s => {
            type_id = GGUFDataType.ggufTypeFloat64;
            is_array = true;
        },
        .bools => {
            type_id = GGUFDataType.ggufTypeBool;
            is_array = true;
        },
        .strs => {
            type_id = GGUFDataType.ggufTypeString;
            is_array = true;
        },
    }

    const type_flag: u32 = if (is_array)
        (1 << 31) | @intFromEnum(type_id)
    else
        @intFromEnum(type_id);

    try writer.writeU32(type_flag);

    if (is_array) {
        const length: u64 = switch (value) {
            .u8s => value.u8s.len,
            .i8s => value.i8s.len,
            .u16s => value.u16s.len,
            .i16s => value.i16s.len,
            .u32s => value.u32s.len,
            .i32s => value.i32s.len,
            .u64s => value.u64s.len,
            .i64s => value.i64s.len,
            .f32s => value.f32s.len,
            .f64s => value.f64s.len,
            .bools => value.bools.len,
            .strs => value.strs.len,
            else => unreachable,
        };
        try writer.writeU64(length);

        switch (value) {
            .u8s => for (value.u8s) |v| try writer.writeU8(v),
            .i8s => for (value.i8s) |v| try writer.writeI8(v),
            .u16s => for (value.u16s) |v| try writer.writeU16(v),
            .i16s => for (value.i16s) |v| try writer.writeI16(v),
            .u32s => for (value.u32s) |v| try writer.writeU32(v),
            .i32s => for (value.i32s) |v| try writer.writeI32(v),
            .u64s => for (value.u64s) |v| try writer.writeU64(v),
            .i64s => for (value.i64s) |v| try writer.writeI64(v),
            .f32s => for (value.f32s) |v| try writer.writeF32(v),
            .f64s => for (value.f64s) |v| try writer.writeF64(v),
            .bools => for (value.bools) |v| try writer.writeBool(v),
            .strs => for (value.strs) |v| try writer.writeString(v),
            else => unreachable,
        }
    } else {
        switch (value) {
            .u8 => try writer.writeU8(value.u8),
            .i8 => try writer.writeI8(value.i8),
            .u16 => try writer.writeU16(value.u16),
            .i16 => try writer.writeI16(value.i16),
            .u32 => try writer.writeU32(value.u32),
            .i32 => try writer.writeI32(value.i32),
            .u64 => try writer.writeU64(value.u64),
            .i64 => try writer.writeI64(value.i64),
            .f32 => try writer.writeF32(value.f32),
            .f64 => try writer.writeF64(value.f64),
            .bool => try writer.writeBool(value.bool),
            .str => try writer.writeString(value.str),
            else => unreachable,
        }
    }
}

pub fn indexOfStringInList(haystack: []const []const u8, needle: []const u8) ?usize {
    for (haystack, 0..) |val, i| {
        if (std.mem.eql(u8, val, needle)) return i;
    }
    return null;
}

pub inline fn tryLoadJson(allocator: std.mem.Allocator, model: registry.ModelInfo, file_path: []const u8) !?std.json.Value {
    const path = try model.localFilePath(model.name, file_path);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const content = try allocator.alloc(u8, file_size);
    _ = try file.readAll(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    return parsed.value;
}

const suffixes = [_][]const u8{ ".weight", ".bias" };

pub fn truncate_name(full: []const u8) []const u8 {
    for (suffixes) |s| {
        if (std.mem.indexOf(u8, full, s)) |idx| {
            return full[0 .. idx + s.len];
        }
    }
    return full; // fallback
}
