const std = @import("std");

pub const Value = union(enum) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    u64: u64,
    i64: i64,
    f32: f32,
    f64: f64,
    str: []const u8,
    strs: []const []const u8,
    bool: bool,
    u8s: []u8,
    i8s: []i8,
    u16s: []u16,
    i16s: []i16,
    u32s: []u32,
    i32s: []i32,
    u64s: []u64,
    i64s: []i64,
    f32s: []f32,
    f64s: []f64,
    bools: []bool,
};

pub const KVMap = struct {
    KV: std.StringHashMap(Value),
    const Self = @This();
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .KV = std.StringHashMap(Value).init(allocator),
        };
    }
    pub fn deinit(self: *Self) void {
        self.KV.deinit();
    }
    pub fn stringValue(self: Self, key: []const u8, default: []const u8) []const u8 {
        if (self.KV.get(key)) |val| {
            if (val == .str) return val.str;
            std.log.warn("Key '{s}' found but not of expected type 'str'", .{key});
        }
        return default;
    }

    pub fn architecture(self: Self) []const u8 {
        return self.stringValue("general.architecture", "unknown");
    }

    pub fn kind(self: Self) []const u8 {
        return self.stringValue("general.type", "unknown");
    }
    pub fn has(self: Self, key: []const u8) bool {
        return self.KV.get(key) != null;
    }

    pub fn ParameterCount(self: Self) u64 {
        return self.keyValue(u64, "general.parameter_count", u64(0));
    }

    fn BlockCount(self: Self) u64 {
        return @as(u64, @intCast((self.Uint("block_count"))));
    }

    fn EmbeddingLength(self: Self) u64 {
        return @as(u64, @intCast((self.Uint("embedding_length"))));
    }

    fn HeadCount(self: Self) u64 {
        return @as(u64, @intCast((self.Uint("attention.head_count"))));
    }

    fn HeadCountKV(self: Self) u64 {
        return @as(u64, @intCast((self.Uint("attention.head_count_kv", 1))));
    }

    fn EmbeddingHeadCount(self: Self) u64 {
        const heads = self.HeadCount();
        if (heads > 0) {
            return self.EmbeddingLength() / heads;
        }

        return 0;
    }

    pub fn EmbeddingHeadCountK(self: Self) u64 {
        return @as(u64, @intCast((self.Uint("attention.key_length", @as(u32, @intCast(self.EmbeddingHeadCount()))))));
    }

    pub fn EmbeddingHeadCountV(self: Self) u64 {
        return @as(u64, @intCast((self.Uint("attention.value_length", @as(u32, @intCast(self.EmbeddingHeadCount()))))));
    }

    fn GQA(self: Self) u64 {
        return self.HeadCount() / self.HeadCountKV();
    }

    fn ContextLength(self: Self) u64 {
        return @as(u64, @intCast(self.Uint("context_length", 0)));
    }

    fn ChatTemplate(self: Self) []const u8 {
        return self.String("tokenizer.chat_template");
    }
    fn String(self: Self, key: []const u8, default_value: ?[]const u8) []const u8 {
        const value = self.keyValue([]const u8, key, default_value orelse "");
        return value;
    }

    pub fn Uint(self: Self, key: []const u8, default_value: ?u32) u32 {
        return self.keyValue(u32, key, default_value orelse 0);
    }

    pub fn Float(self: Self, key: []const u8, default_value: f32) f32 {
        return self.keyValue(f32, key, default_value);
    }

    pub fn Bool(self: Self, key: []const u8, default_value: bool) bool {
        return self.keyValue(bool, key, default_value);
    }

    pub fn Strings(self: Self, key: []const u8, default_value: []const []const u8) []const []const u8 {
        return self.keyValue([]const []const u8, key, default_value);
    }

    pub fn Ints(self: Self, key: []const u8, default_value: []i32) []i32 {
        return self.keyValue([]i32, key, default_value);
    }

    pub fn Uints(self: Self, key: []const u8, default_value: []u32) []u32 {
        return self.keyValue([]u32, key, default_value);
    }

    pub fn Floats(self: Self, key: []const u8, default_value: []f32) []f32 {
        return self.keyValue([]f32, key, default_value);
    }

    pub inline fn keyValue(self: Self, comptime T: type, key: []const u8, default_value: T) T {
        var final_key = key;

        if (!std.mem.startsWith(u8, key, "tokenizer.") and !std.mem.startsWith(u8, key, "general.")) {
            const arch = self.architecture();
            const allocator = std.heap.page_allocator;

            var buffer = std.ArrayList(u8).init(allocator);
            defer buffer.deinit();

            _ = buffer.appendSlice(arch) catch return default_value;
            _ = buffer.appendSlice(".") catch return default_value;
            _ = buffer.appendSlice(key) catch return default_value;
            final_key = buffer.toOwnedSlice() catch return default_value;
        }

        if (self.KV.get(final_key)) |val| {
            switch (T) {
                i32 => if (val == .i32) return val.i32,
                u32 => if (val == .u32) return val.u32,
                i64 => if (val == .i64) return val.i64,
                u64 => if (val == .u64) return val.u64,
                f32 => if (val == .f32) return val.f32,
                []const u8 => if (val == .str) return val.str,
                []const []const u8 => if (val == .strs) return val.strs,
                bool => if (val == .bool) return val.bool,
                []u32 => if (val == .u32s) return val.u32s,
                []i32 => if (val == .i32s) return val.i32s,
                []f32 => if (val == .f32s) return val.f32s,
                else => {},
            }
        }

        std.log.warn("Config missing key: {s}, defaulting to: {any}", .{ final_key, default_value });

        return default_value;
    }
};
