const std = @import("std");
const ggufPadding = @import("gguf.zig").ggufPadding;
const GGUFDataType = @import("types.zig").GGUFDataType;
const Value = @import("KV.zig").Value;
pub const GGUFWriter = struct {
    writer: std.io.AnyWriter,
    position: usize, // Tracks current write position for alignment

    const Self = @This();

    pub fn init(writer: std.io.AnyWriter) Self {
        return Self{
            .writer = writer,
            .position = 0,
        };
    }

    pub fn advance(self: *Self, n: usize) void {
        self.position += n;
    }

    pub fn writeU8(self: *Self, v: u8) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(1);
    }
    pub fn writeU16(self: *Self, v: u16) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(2);
    }

    pub fn writeU32(self: *Self, v: u32) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(4);
    }

    pub fn writeU64(self: *Self, v: u64) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(8);
    }

    pub fn writeI8(self: *Self, v: i8) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(1);
    }

    pub fn writeI16(self: *Self, v: i16) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(2);
    }

    pub fn writeI32(self: *Self, v: i32) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(4);
    }

    pub fn writeI64(self: *Self, v: i64) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(8);
    }

    pub fn writeF32(self: *Self, v: f32) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(4);
    }

    pub fn writeF64(self: *Self, v: f64) !void {
        try self.writer.writeAll(&std.mem.toBytes(v));
        self.advance(8);
    }

    pub fn writeBool(self: *Self, v: bool) !void {
        const b: u8 = if (v) 1 else 0;
        try self.writeU8(b);
    }

    pub fn writeString(self: *Self, str: []const u8) !void {
        try self.writeU64(@intCast(str.len));
        try self.writer.writeAll(str);
        self.advance(str.len);
    }

    pub fn writeArray(self: *Self, comptime T: type, data: []const T) !void {
        try self.writeU64(@intCast(data.len));
        for (data) |v| {
            try self.writer.writeAll(&std.mem.toBytes(v));
            self.advance(@sizeOf(T));
        }
    }
    pub fn writeGGUFArray(
        self: *GGUFWriter,
        comptime T: type,
        type_id: u32,
        data: []const T,
    ) !void {
        try self.writeU32(type_id); // Type header
        try self.writeU64(@intCast(data.len)); // Length

        for (data) |item| {
            try self.writer.writeAll(&std.mem.toBytes(item));
            self.advance(@sizeOf(T));
        }
    }
    pub fn writeGGUFStringArray(
        self: *GGUFWriter,
        strs: [][]const u8,
    ) !void {
        try self.writeU32(@intFromEnum(GGUFDataType.ggufTypeString)); // Type ID for string
        try self.writeU64(@intCast(strs.len)); // Length

        for (strs) |s| {
            try self.writeString(s); // String writer already handles length prefix
        }
    }
    pub fn writeGGUFBoolArray(
        self: *GGUFWriter,
        data: []const bool,
    ) !void {
        try self.writeU32(@intFromEnum(GGUFDataType.ggufTypeBool)); // Bool type id
        try self.writeU64(@intCast(data.len));

        for (data) |b| {
            try self.writeBool(b);
        }
    }
    pub fn writeGGUFArrayAuto(self: *GGUFWriter, value: Value) !void {
        switch (value) {
            .u8s => try self.writeGGUFArray(u8, @intFromEnum(GGUFDataType.ggufTypeUint8), value.u8s),
            .i8s => try self.writeGGUFArray(i8, @intFromEnum(GGUFDataType.ggufTypeInt8), value.i8s),
            .u16s => try self.writeGGUFArray(u16, @intFromEnum(GGUFDataType.ggufTypeUint16), value.u16s),
            .i16s => try self.writeGGUFArray(i16, @intFromEnum(GGUFDataType.ggufTypeInt16), value.i16s),
            .u32s => try self.writeGGUFArray(u32, @intFromEnum(GGUFDataType.ggufTypeUint32), value.u32s),
            .i32s => try self.writeGGUFArray(i32, @intFromEnum(GGUFDataType.ggufTypeInt32), value.i32s),
            .u64s => try self.writeGGUFArray(u64, @intFromEnum(GGUFDataType.ggufTypeUint64), value.u64s),
            .i64s => try self.writeGGUFArray(i64, @intFromEnum(GGUFDataType.ggufTypeInt64), value.i64s),
            .f32s => try self.writeGGUFArray(f32, @intFromEnum(GGUFDataType.ggufTypeFloat32), value.f32s),
            .f64s => try self.writeGGUFArray(f64, @intFromEnum(GGUFDataType.ggufTypeFloat64), value.f64s),
            .bools => try self.writeGGUFBoolArray(value.bools),
            .strs => try self.writeGGUFStringArray(value.strs),
            else => return error.UnsupportedArrayType,
        }
    }
    pub fn writePadding(self: *Self, alignment: u64) !void {
        const offset = self.position;
        const pad = ggufPadding(offset, alignment);
        if (pad == 0) return;

        var remaining: usize = @as(usize, @intCast(pad));
        const zeros = [_]u8{0} ** 256;

        while (remaining > 0) : (remaining -= @min(remaining, 256)) {
            const chunk = @min(remaining, 256);
            try self.writer.writeAll(zeros[0..chunk]);
            self.advance(chunk);
        }
    }

    // pub fn writePadding(self: *Self, alignment: u64) !void {
    //     const offset = @as(u64, @intCast(self.position));
    //     const pad = ggufPadding(offset, alignment);
    //     if (pad > 0) {
    //         const zeroes = [_]u8{0} ** 256;
    //         try self.writer.writeAll(zeroes[0..@intCast(pad)]);
    //         self.advance(@intCast(pad));
    //     }
    // }
    pub fn writeTokenList(self: *GGUFWriter, tokens: [][]const u8) !void {
        // write array length
        try self.writeU64(@as(u64, @intCast(tokens.len)));
        // for each token string: writeString()
        for (tokens) |token| {
            try self.writeString(token);
        }
    }

    pub fn writeTokenScores(self: *GGUFWriter, scores: []f32) !void {
        try self.writeU64(@as(u64, @intCast(scores.len)));
        for (scores) |score| {
            try self.writeF32(score);
        }
    }

    pub fn writeTokenTypes(self: *GGUFWriter, types: []i32) !void {
        try self.writeU64(@as(u64, @intCast(types.len)));
        for (types) |typ| {
            try self.writeI32(typ);
        }
    }
};
