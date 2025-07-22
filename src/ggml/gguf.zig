const std = @import("std");
const registry = @import("../registry/model_registry.zig");
const ggml = @import("ggml.zig");
const GGUFDataType = @import("types.zig").GGUFDataType;
const Tensors = @import("tensors.zig").Tensors;
const Tensor = @import("tensor.zig").Tensor;
const KV = @import("KV.zig").KVMap;
const Value = @import("KV.zig").Value;
const FileReader = @import("reader.zig").FileReader;
const native_endian = @import("builtin").target.cpu.arch.endian();

const GGUF = struct {
    container: *ContainerGGUF,
    kv: KV,
    tensors: Tensors,
    parameters: ?u64 = undefined,
    tensorOffset: ?u64 = undefined,
    alignment: ?u32 = undefined,

    const Self = @This();
    pub fn init(container: *ContainerGGUF, allocator: std.mem.Allocator) Self {
        return Self{
            .container = container,
            .kv = KV.init(allocator),
            .tensors = Tensors.init(allocator),
            .parameters = 0,
            .alignment = 0,
            .tensorOffset = null,
        };
    }
    pub fn getKV(self: *Self) KV {
        return self.kv;
    }
    pub fn getTensors(self: *Self) Tensors {
        return self.tensors;
    }

    pub fn numTensor(self: *Self) u64 {
        switch (self.container.version) {
            1 => return @as(u64, @intCast(self.container.version_data.v1.num_tensor)),
            2 => return self.container.version_data.v2.num_tensor,
            else => return self.container.version_data.v3.num_tensor,
        }
    }
    pub fn numKV(self: *Self) u64 {
        switch (self.container.version) {
            1 => return @as(u64, @intCast(self.container.version_data.v1.num_kv)),
            2 => return self.container.version_data.v2.num_kv,
            else => return self.container.version_data.v3.num_kv,
        }
    }

    pub fn decode(llm: *Self, reader: *FileReader, allocator: std.mem.Allocator) !void {
        // decode key-values
        var i: usize = 0;
        const n_kv = llm.numKV();
        try llm.kv.KV.ensureTotalCapacity(@as(u32, @intCast(n_kv)));

        while (@as(u64, @intCast(i)) < n_kv) : (i += 1) {
            const k = try readGGUFStringFixed(reader, llm, allocator);
            const type_id_u32 = try readGGUF(u32, reader, llm);
            std.debug.print("Key: {s}\n", .{k});
            std.debug.print("Type ID: {} (0x{x})\n", .{ type_id_u32, type_id_u32 });
            std.debug.print("Next pos: {}\n", .{reader.pos});
            const type_id = try std.meta.intToEnum(GGUFDataType, type_id_u32);
            var v: Value = undefined;
            switch (type_id) {
                .ggufTypeUint8 => v = .{ .u8 = try readGGUF(u8, reader, llm) },
                .ggufTypeInt8 => v = .{ .i8 = try readGGUF(i8, reader, llm) },
                .ggufTypeUint16 => v = .{ .u16 = try readGGUF(u16, reader, llm) },
                .ggufTypeInt16 => v = .{ .i16 = try readGGUF(i16, reader, llm) },
                .ggufTypeUint32 => v = .{ .u32 = try readGGUF(u32, reader, llm) },
                .ggufTypeInt32 => v = .{ .i32 = try readGGUF(i32, reader, llm) },
                .ggufTypeUint64 => v = .{ .u64 = try readGGUF(u64, reader, llm) },
                .ggufTypeInt64 => v = .{ .i64 = try readGGUF(i64, reader, llm) },
                .ggufTypeFloat32 => v = .{ .f32 = try readGGUF(f32, reader, llm) },
                .ggufTypeFloat64 => v = .{ .f64 = try readGGUF(f64, reader, llm) },
                .ggufTypeBool => v = .{ .bool = try readGGUF(bool, reader, llm) },
                .ggufTypeString => v = .{ .str = try readGGUFStringFixed(reader, llm, allocator) },
                .ggufTypeArray => v = try readGGUFArray(llm, reader, allocator),
            }

            try llm.kv.KV.put(k, v);
            std.debug.print("Total KVs decoded: {}/{}\n", .{ i, n_kv });
        }

        // decode tensors
        var t: usize = 0;
        const n_tensors = llm.numTensor();
        while (@as(u64, @intCast(t)) < n_tensors) : (t += 1) {
            const name = try readGGUFStringFixed(reader, llm, allocator);
            const dims = try readGGUF(u32, reader, llm);

            const shape = try allocator.alloc(u64, dims);
            for (shape) |*s| {
                s.* = try readGGUF(u64, reader, llm);
            }

            const kind = try readGGUF(u32, reader, llm);
            const offset = try readGGUF(u64, reader, llm);
            std.debug.print("Key: {s}\n", .{name});

            var tensor = Tensor{
                .name = name,
                .kind = kind,
                .offset = offset,
                .shape = shape,
            };

            try llm.tensors.items.append(tensor);
            llm.parameters.? += tensor.elements();
        }
        std.debug.print("Total Parameter Count: {d}\n", .{llm.parameters.?});
        try llm.kv.KV.put("general.parameter_count", .{ .u64 = llm.parameters.? });

        const alignment = llm.kv.Uint("general.alignment", 32);
        llm.alignment = alignment;
        const offset: u64 = @as(u64, @intCast(reader.pos));
        const padding = ggufPadding(offset, alignment);
        llm.tensorOffset = offset + padding;

        for (llm.tensors.items.items) |tensor| {
            const cur_offset = @as(u64, @intCast(reader.pos));
            const pad = ggufPadding(cur_offset, alignment);
            _ = try reader.readBytes(@as(usize, @intCast(pad)));
            _ = try reader.readBytes(@as(usize, @intCast(tensor.size())));
        }
    }
};

pub fn ggufPadding(offset: u64, alignment: u64) u64 {
    return (alignment - (offset % alignment)) % alignment;
}
pub fn readGGUF(comptime T: type, reader: *FileReader, llm: *const GGUF) !T {
    const byte_count = @sizeOf(T);
    const bytes = try reader.readBytes(byte_count);
    const ti = @typeInfo(T);
    switch (ti) {
        .int, .comptime_int => {
            var value: T = std.mem.bytesToValue(T, bytes);
            if (llm.container.byte_order != native_endian) {
                value = @byteSwap(value);
            }
            return value;
        },
        .float, .comptime_float => {
            const int_type = switch (@sizeOf(T)) {
                4 => u32,
                8 => u64,
                else => return error.UnsupportedFloatSize,
            };
            var int_value = std.mem.bytesToValue(int_type, bytes);
            if (llm.container.byte_order != native_endian) {
                int_value = @byteSwap(int_value);
            }
            return @bitCast(int_value);
        },
        .bool => {
            if (byte_count != 1) return error.InvalidBoolSize;
            return @as(T, bytes[0] != 0);
        },
        else => return error.UnsupportedType,
    }
}

pub fn readGGUFStringFixed(reader: *FileReader, llm: *GGUF, allocator: std.mem.Allocator) ![]u8 {
    if (llm.container.version == 1) {
        return try readGGUFV1String(reader, llm, allocator);
    }

    const len_bytes = try reader.readBytes(8);
    var len_buf: [8]u8 = undefined;
    @memcpy(&len_buf, len_bytes);
    const length = std.mem.readInt(u64, &len_buf, llm.container.byte_order);

    if (length > 1_000_000) return error.InvalidStringLength;

    const len_usize = @as(usize, @intCast(length));

    const raw = try reader.readBytes(len_usize);

    const buf = try allocator.alloc(u8, len_usize);
    @memcpy(buf, raw);

    return buf;
}

pub fn readGGUFV1String(reader: *FileReader, llm: *GGUF, allocator: std.mem.Allocator) ![]u8 {
    const len_bytes = try reader.readBytes(8);
    var len_buf: [8]u8 = undefined;
    @memcpy(&len_buf, len_bytes);
    const length = std.mem.readInt(u64, &len_buf, llm.container.byte_order);

    const len_usize = @as(usize, @intCast(length));
    const buf = try allocator.alloc(u8, len_usize);
    const raw = try reader.readBytes(len_usize);
    @memcpy(buf, raw);

    // gguf v1 strings are null-terminated
    if (buf.len == 0 or buf[buf.len - 1] != 0) {
        return error.InvalidV1String;
    }

    return buf[0 .. buf.len - 1];
}

pub fn Array(comptime T: type) type {
    return struct {
        size: usize,
        values: ?[]T,

        pub fn newArray(size: usize, max_size: usize, allocator: std.mem.Allocator) !*@This() {
            var a = try allocator.create(@This());
            a.* = .{
                .size = size,
                .values = null,
            };

            if (max_size < 0 or size <= @as(usize, @intCast(max_size))) {
                a.values = try allocator.alloc(T, size);
            }

            return a;
        }
    };
}

pub fn readGGUFArray(llm: *GGUF, reader: *FileReader, allocator: std.mem.Allocator) !Value {
    const type_id_u32 = try readGGUF(u32, reader, llm);
    const n = try readGGUF(u64, reader, llm);
    const type_id = @as(GGUFDataType, @enumFromInt(type_id_u32));
    const size = @as(usize, @intCast(n));

    return switch (type_id) {
        .ggufTypeUint8 => Value{ .u8s = try readGGUFArrayData(u8, llm, reader, try allocator.alloc(u8, size)) },
        .ggufTypeInt8 => Value{ .i8s = try readGGUFArrayData(i8, llm, reader, try allocator.alloc(i8, size)) },
        .ggufTypeUint16 => Value{ .u16s = try readGGUFArrayData(u16, llm, reader, try allocator.alloc(u16, size)) },
        .ggufTypeInt16 => Value{ .i16s = try readGGUFArrayData(i16, llm, reader, try allocator.alloc(i16, size)) },
        .ggufTypeUint32 => Value{ .u32s = try readGGUFArrayData(u32, llm, reader, try allocator.alloc(u32, size)) },
        .ggufTypeInt32 => Value{ .i32s = try readGGUFArrayData(i32, llm, reader, try allocator.alloc(i32, size)) },
        .ggufTypeUint64 => Value{ .u64s = try readGGUFArrayData(u64, llm, reader, try allocator.alloc(u64, size)) },
        .ggufTypeInt64 => Value{ .i64s = try readGGUFArrayData(i64, llm, reader, try allocator.alloc(i64, size)) },
        .ggufTypeFloat32 => Value{ .f32s = try readGGUFArrayData(f32, llm, reader, try allocator.alloc(f32, size)) },
        .ggufTypeFloat64 => Value{ .f64s = try readGGUFArrayData(f64, llm, reader, try allocator.alloc(f64, size)) },
        .ggufTypeBool => {
            const bools = try allocator.alloc(bool, size);
            return Value{ .bools = try readGGUFArrayData(bool, llm, reader, bools) };
        },
        .ggufTypeString => {
            const strs = try allocator.alloc([]const u8, size);
            for (0.., strs) |i, _| {
                strs[i] = if (llm.container.version == 1)
                    try readGGUFV1String(reader, llm, allocator)
                else
                    try readGGUFStringFixed(reader, llm, allocator);
            }
            return Value{ .strs = strs };
        },
        else => return error.InvalidArrayType,
    };
}

pub fn readGGUFArrayData(comptime T: type, llm: *GGUF, reader: *FileReader, buf: []T) ![]T {
    for (buf) |*item| {
        item.* = try readGGUF(T, reader, llm);
    }
    return buf;
}

pub fn describe(model_name: []const u8, allocator: std.mem.Allocator) !void {
    const model = try registry.findModelErrorless(model_name);
    const found_model = model.?;

    std.debug.print("Loading model: {s}\n", .{found_model.name});

    const buffer = try found_model.loadGGUFModelBuffer(allocator);
    defer allocator.free(buffer);
    const description = try ggml.GGML.describeGGUF(buffer, 4096, allocator);
    ggml.printDescriptor(description);
}

pub fn read(model_name: []const u8, allocator: std.mem.Allocator) !void {
    const model = try registry.findModelErrorless(model_name);
    const found_model = model.?;

    std.debug.print("Loading model: {s}\n", .{found_model.name});

    const buffer = try found_model.loadGGUFModelBuffer(allocator);
    defer allocator.free(buffer);
    const mod = try ggml.GGML.decode(buffer, 4096, allocator);

    std.debug.print("Model loaded: {s}\n", .{mod.Name});
    // TODO
    // for (mod.Tensors.items.items) |tensor|{
    //     tensor.name
    //     tensor.tensorType()
    // }

}
const VersionData = union(enum) {
    v1: struct {
        num_tensor: u32,
        num_kv: u32,
    },
    v2: struct {
        num_tensor: u64,
        num_kv: u64,
    },
    v3: struct {
        num_tensor: u64,
        num_kv: u64,
    },
};

pub const ContainerGGUF = struct {
    version: u32,
    byte_order: std.builtin.Endian,
    version_data: VersionData,
    max_array_size: usize,

    const Self = @This();
    pub fn init(byte_order: std.builtin.Endian, max_array_size: usize) Self {
        return Self{
            .byte_order = byte_order,
            .version = 0,
            .version_data = VersionData{ .v3 = .{ .num_tensor = 0, .num_kv = 0 } },
            .max_array_size = max_array_size,
        };
    }
    pub fn Name() []const u8 {
        return "gguf";
    }
    pub fn decode(self: *ContainerGGUF, reader: *FileReader, allocator: std.mem.Allocator) !GGUF {
        self.version = try reader.readU32();

        switch (self.version) {
            1 => {
                self.version_data.v1.num_tensor = switch (self.byte_order) {
                    .little => std.mem.littleToNative(u32, try reader.readU32()),
                    .big => std.mem.bigToNative(u32, try reader.readU32()),
                };
                self.version_data.v1.num_kv = switch (self.byte_order) {
                    .little => std.mem.littleToNative(u32, try reader.readU32()),
                    .big => std.mem.bigToNative(u32, try reader.readU32()),
                };
            },
            2 => {
                self.version_data.v2.num_tensor = switch (self.byte_order) {
                    .little => std.mem.littleToNative(u64, try reader.readU64()),
                    .big => std.mem.bigToNative(u64, try reader.readU64()),
                };
                self.version_data.v2.num_kv = switch (self.byte_order) {
                    .little => std.mem.littleToNative(u64, try reader.readU64()),
                    .big => std.mem.bigToNative(u64, try reader.readU64()),
                };
            },
            else => {
                self.version_data.v3.num_tensor = switch (self.byte_order) {
                    .little => std.mem.littleToNative(u64, try reader.readU64()),
                    .big => std.mem.bigToNative(u64, try reader.readU64()),
                };
                self.version_data.v3.num_kv = switch (self.byte_order) {
                    .little => std.mem.littleToNative(u64, try reader.readU64()),
                    .big => std.mem.bigToNative(u64, try reader.readU64()),
                };
            },
        }
        const gguf_size = @sizeOf(GGUF);
        const gguf_alignment = @alignOf(GGUF);
        std.debug.print("GGUF size: {} bytes\n", .{gguf_size});
        std.debug.print("GGUF alignment: {} bytes\n", .{gguf_alignment});
        var model = GGUF.init(self, allocator);
        try model.decode(reader, allocator);
        return model;
    }
};

pub fn newGGUF(container: *ContainerGGUF, allocator: std.mem.Allocator) *GGUF {
    return &GGUF{
        .container = container,
        .kv = KV.init(allocator),
    };
}
