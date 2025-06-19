const std = @import("std");

pub const FileReader = struct {
    buf: []const u8,
    pos: usize,

    const Self = @This();
    pub fn init(buf: []const u8) Self {
        return Self{
            .buf = buf,
            .pos = 0,
        };
    }

    fn ensure(self: *Self, n: usize) !void {
        if (self.pos + n > self.buf.len) return error.EndOfFile;
    }

    pub fn readBytes(self: *Self, n: usize) ![]const u8 {
        try self.ensure(n);
        const slice = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return slice;
    }
    pub fn readU8(self: *Self) !u8 {
        const bytes = try self.readBytes(1);
        return std.mem.bytesToValue(u8, bytes);
    }
    pub fn readU32(self: *Self) !u32 {
        const bytes = try self.readBytes(4);
        return std.mem.bytesToValue(u32, bytes);
    }
    pub fn readU64(self: *FileReader) !u64 {
        const bytes = try self.readBytes(8);
        return std.mem.bytesToValue(u64, bytes);
    }

    pub fn readString(self: *Self, len: usize) ![]const u8 {
        const bytes = try self.readBytes(len);
        return bytes;
    }
};
