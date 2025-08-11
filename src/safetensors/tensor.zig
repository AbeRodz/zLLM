const std = @import("std");
const registry = @import("../registry/model_registry.zig");
const FileReader = @import("../ggml/reader.zig").FileReader;
const Value = @import("../ggml/KV.zig").Value;
pub const SafeTensors = struct {
    metadata: Metadata,
    data: []u8,
};

const TensorEntry = struct {
    name: []const u8,
    info: TensorInfo,
};

const OffsetEntry = struct {
    name: []const u8,
    index: usize,
};
pub const Metadata = struct {
    model_name: []const u8,
    metadata: std.StringArrayHashMap(Value),
    tensors: std.ArrayList(TensorInfo),
    index_map: std.StringHashMap(usize),
    pub fn init(
        allocator: std.mem.Allocator,
        model_name: []const u8,
        meta_opt: ?std.StringArrayHashMap(Value),
        tensor_entries: []TensorEntry,
    ) !Metadata {
        var index_map = std.StringHashMap(usize).init(allocator);
        var tensors = std.ArrayList(TensorInfo).init(allocator);

        for (0.., tensor_entries) |i, entry| {
            try index_map.put(entry.name, i);
            try tensors.append(entry.info);
        }

        const metadata = Metadata{
            .model_name = model_name,
            .metadata = meta_opt orelse std.StringArrayHashMap(Value).init(allocator),
            .tensors = tensors,
            .index_map = index_map,
        };

        return metadata;
    }
    pub fn put(self: *Metadata, allocator: std.mem.Allocator, key_suffix: []const u8, value: Value) !void {
        const full_key_tmp = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ self.model_name, key_suffix });
        // Allocate a buffer just large enough to hold the string and copy it
        const full_key = try allocator.alloc(u8, full_key_tmp.len);
        @memcpy(full_key, full_key_tmp);
        allocator.free(full_key_tmp); // free tmp buffer after copying

        // Insert the persistent key string
        try self.metadata.put(full_key, value);
        std.debug.print("Inserting key: {s}\n", .{full_key});
    }
    pub fn get(self: *Metadata, allocator: std.mem.Allocator, key_suffix: []const u8) ?Value {
        // Build the full key just like in `put`
        const full_key_tmp = std.fmt.allocPrint(allocator, "{s}.{s}", .{ self.model_name, key_suffix }) catch return null;
        defer allocator.free(full_key_tmp);

        return self.metadata.get(full_key_tmp);
    }

    pub fn validate(self: *const Metadata) !usize {
        var start: u64 = 0;

        for (self.tensors.items, 0..) |tensor, i| {
            const s = tensor.data_offsets.start;
            const e = tensor.data_offsets.end;

            if (s != start or e < s) {
                var tensor_name = "no_tensor";
                var iter = self.index_map.iterator();
                while (iter.next()) |entry| {
                    if (entry.value_ptr.* == i) {
                        tensor_name = entry.key_ptr.*;
                        break;
                    }
                }
                std.debug.print("Invalid offset in tensor: {s}\n", .{tensor_name});
                return error.InvalidOffset;
            }

            start = e;

            var nelements: usize = 1;
            for (tensor.shape) |dim| {
                nelements = std.math.mul(usize, nelements, @as(usize, @intCast(dim))) catch {
                    return error.ValidationOverflow;
                };
            }

            const nbits = std.math.mul(usize, nelements, info.bitsize()) catch {
                return error.ValidationOverflow;
            };

            if (nbits % 8 != 0) return error.MisalignedSlice;

            const size = nbits / 8;
            const actual_size = @as(usize, @intCast(e - s));
            if (actual_size != size) return error.TensorInvalidInfo;
        }

        return @as(usize, @intCast(start));
    }
    pub fn info(self: *Metadata, name: []const u8) ?*const TensorInfo {
        const index = self.index_map.get(name) orelse return null;
        return &self.tensors.items[index.*];
    }
    pub fn getTensors(self: *Metadata, allocator: std.mem.Allocator) !std.StringHashMap(*const TensorInfo) {
        var map = std.StringHashMap(*const TensorInfo).init(allocator);

        var iter = self.index_map.iterator();
        while (iter.next()) |entry| {
            const name = entry.key_ptr.*;
            const index = entry.value_ptr.*;
            try map.put(name, &self.tensors.items[index.*]);
        }

        return map;
    }
    pub fn offsetKeys(self: Metadata, allocator: std.mem.Allocator) ![][]const u8 {
        var kvs = try allocator.alloc(OffsetEntry, self.index_map.count());

        var i: usize = 0;
        var iter = self.index_map.iterator();
        while (iter.next()) |entry| {
            kvs[i] = OffsetEntry{ .name = entry.key_ptr.*, .index = entry.value_ptr.* };
            i += 1;
        }

        std.mem.sort(
            OffsetEntry,
            kvs,
            {}, // context
            struct {
                pub fn lessThan(_: void, lhs: OffsetEntry, rhs: OffsetEntry) bool {
                    return std.mem.lessThan(u8, lhs.name, rhs.name);
                }
            }.lessThan,
        );

        var result = try allocator.alloc([]const u8, kvs.len);
        for (kvs, 0..) |kv, j| {
            result[j] = kv.name;
        }

        return result;
    }
    pub fn dataLen(self: *Metadata) usize {
        if (self.tensors.items.len == 0) return 0;
        return @as(usize, @intCast(self.tensors.items[self.tensors.items.len - 1].data_offsets.end));
    }
    pub fn getMetadata(self: *Metadata) *const std.StringHashMap([]const u8) {
        return &self.metadata;
    }
};

pub const TensorInfo = struct {
    dtype: []const u8,
    shape: []u64,
    data_offsets: struct {
        start: u64,
        end: u64,
    },
    pub fn bitsize(self: *const TensorInfo) usize {
        if (std.mem.eql(u8, self.dtype, "F4")) return 4;
        if (std.mem.eql(u8, self.dtype, "F6_E3M2")) return 6;
        if (std.mem.eql(u8, self.dtype, "F6_E2M3")) return 6;
        if (std.mem.eql(u8, self.dtype, "BOOL")) return 8;
        if (std.mem.eql(u8, self.dtype, "U8")) return 8;
        if (std.mem.eql(u8, self.dtype, "I8")) return 8;
        if (std.mem.eql(u8, self.dtype, "F8_E5M2")) return 8;
        if (std.mem.eql(u8, self.dtype, "F8_E4M3")) return 8;
        if (std.mem.eql(u8, self.dtype, "F8_E8M0")) return 8;
        if (std.mem.eql(u8, self.dtype, "I16")) return 16;
        if (std.mem.eql(u8, self.dtype, "U16")) return 16;
        if (std.mem.eql(u8, self.dtype, "F16")) return 16;
        if (std.mem.eql(u8, self.dtype, "BF16")) return 16;
        if (std.mem.eql(u8, self.dtype, "I32")) return 32;
        if (std.mem.eql(u8, self.dtype, "U32")) return 32;
        if (std.mem.eql(u8, self.dtype, "F32")) return 32;
        if (std.mem.eql(u8, self.dtype, "I64")) return 64;
        if (std.mem.eql(u8, self.dtype, "U64")) return 64;
        if (std.mem.eql(u8, self.dtype, "F64")) return 64;

        return 0; // Unknown dtype
    }
};
pub fn parseSafetensorsFromBufferV2(allocator: std.mem.Allocator, model_name: []const u8, buffer: []const u8) !Metadata {
    var reader = FileReader.init(buffer);

    // Read header length (first 8 bytes as little-endian u64)
    const len_bytes = try reader.readBytes(8);
    var len_buf: [8]u8 = undefined;
    @memcpy(&len_buf, len_bytes);
    const length = std.mem.readInt(u64, &len_buf, .little);
    std.debug.print("Header JSON length (from file): {}\n", .{length});
    const header_len: usize = 8 + @as(usize, @intCast(length));
    std.debug.print("Total header bytes (including 8-byte size): {}\n", .{header_len});

    const header_raw = try reader.readBytes(@as(usize, @intCast(length)));
    const header_buf = try allocator.alloc(u8, header_raw.len);
    @memcpy(header_buf, header_raw);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidHeader;

    var meta_map = std.StringArrayHashMap(Value).init(allocator);
    var tensor_entries = std.ArrayList(TensorEntry).init(allocator);

    var iter = root.object.iterator();
    while (iter.next()) |kv| {
        const name = kv.key_ptr.*;
        const tensor_obj = kv.value_ptr.*;

        if (std.mem.eql(u8, name, "__metadata__")) {
            if (tensor_obj != .object) continue;
            var meta_iter = tensor_obj.object.iterator();
            while (meta_iter.next()) |meta_kv| {
                const key = meta_kv.key_ptr.*;
                const value = meta_kv.value_ptr.*;
                const val = switch (value) {
                    .string => Value{ .str = value.string },
                    .integer => Value{ .i64 = value.integer },
                    .float => Value{ .f64 = value.float },
                    .bool => Value{ .bool = value.bool },
                    else => {
                        std.debug.print("⚠️ Skipping unsupported metadata value for key {s}\n", .{key});
                        continue;
                    },
                };
                try meta_map.put(key, val);
            }
            continue;
        }

        if (tensor_obj != .object) continue;
        const fields = tensor_obj.object;

        const dtype = fields.get("dtype").?.string;
        const shape_val = fields.get("shape").?.array;
        const offsets_val = fields.get("data_offsets").?.array;

        const valid_dtypes = [_][]const u8{ "F32", "F16", "BF16", "I64", "I32", "U8" };
        var dtype_valid = false;
        for (valid_dtypes) |t| {
            if (std.mem.eql(u8, t, dtype)) {
                dtype_valid = true;
                break;
            }
        }
        if (!dtype_valid) {
            std.debug.print("❌ Unknown dtype: {s}\n", .{dtype});
            return error.InvalidDType;
        }

        if (offsets_val.items.len != 2) return error.InvalidOffsets;
        const raw_start = offsets_val.items[0].integer;
        const raw_end = offsets_val.items[1].integer;

        // Validate raw offset values BEFORE computing final offsets
        if (raw_end < raw_start) {
            std.debug.print("❌ Invalid raw offset: start={}, end={}\n", .{ raw_start, raw_end });
            return error.InvalidOffset;
        }
        const start_offset = @as(u64, @intCast(raw_start)) + header_len;
        const end_offset = @as(u64, @intCast(raw_end)) + header_len;

        if (end_offset > buffer.len or start_offset > end_offset) {
            std.debug.print("❌ Invalid computed offset for {s}: start={}, end={}, file.len={}\n", .{ name, start_offset, end_offset, buffer.len });
            return error.InvalidOffset;
        }
        std.debug.print(
            "Tensor: {s} | raw_start={}, raw_end={} | computed offsets: start={}, end={}, buffer.len={}\n",
            .{ name, raw_start, raw_end, start_offset, end_offset, buffer.len },
        );

        //const start_offset = raw_start + @as(u64, header_len);
        //const end_offset = raw_end + @as(u64, header_len);

        if (end_offset > buffer.len or start_offset > end_offset) {
            std.debug.print("❌ Invalid offset for {s}: start={}, end={}, file.len={}\n", .{ name, start_offset, end_offset, buffer.len });
            return error.InvalidOffset;
        }

        var shape = try allocator.alloc(u64, shape_val.items.len);
        for (shape_val.items, 0..) |dim, i| {
            if (dim != .integer) {
                std.debug.print("❌ Invalid shape dimension (not integer) at index {d}\n", .{i});
                return error.InvalidShape;
            }

            const dim_val = dim.integer;
            if (dim_val < 0 or dim_val > 100_000_000) {
                std.debug.print("❌ Suspicious dimension size at index {d}: {}\n", .{ i, dim_val });
                return error.InvalidShape;
            }

            shape[i] = @as(u64, @intCast(dim_val));
        }

        const tensor = TensorInfo{
            .shape = shape,
            .dtype = dtype,
            .data_offsets = .{
                .start = start_offset,
                .end = end_offset,
            },
        };

        try tensor_entries.append(TensorEntry{ .name = name, .info = tensor });
    }

    const metadata = try Metadata.init(allocator, model_name, meta_map, tensor_entries.items);
    std.debug.print("✅ Safetensors parsing complete.\n", .{});
    return metadata;
}

pub fn parseSafetensorsFromBuffer(allocator: std.mem.Allocator, model_name: []const u8, buffer: []const u8) !Metadata {
    var reader = FileReader.init(buffer);

    // Read header length (first 4 bytes as little-endian u32)
    const len_bytes = try reader.readBytes(8);
    var len_buf: [8]u8 = undefined;
    @memcpy(&len_buf, len_bytes);

    const length = std.mem.readInt(u64, &len_buf, .little);
    const len_usize = @as(usize, @intCast(length));

    const raw = try reader.readBytes(len_usize);

    const buf = try allocator.alloc(u8, len_usize);
    @memcpy(buf, raw);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidHeader;
    var meta_map = std.StringArrayHashMap(Value).init(allocator);
    // Iterate over tensor entries
    var iter = root.object.iterator();

    var tensor_entries = std.ArrayList(TensorEntry).init(allocator);

    while (iter.next()) |kv| {
        const name = kv.key_ptr.*;
        const tensor_obj = kv.value_ptr.*;
        // if (std.mem.eql(u8, name, "__metadata__")) {
        //     continue;
        // }
        if (std.mem.eql(u8, name, "__metadata__")) {
            if (tensor_obj != .object) continue;
            var meta_iter = tensor_obj.object.iterator();
            while (meta_iter.next()) |meta_kv| {
                const key = meta_kv.key_ptr.*;
                if (std.mem.eql(u8, key, "format")) {
                    continue;
                }
                const value = meta_kv.value_ptr.*;
                std.debug.print("Metadata key: {s}\n", .{key});

                const val = switch (value) {
                    .string => Value{ .str = value.string },
                    .integer => Value{ .i64 = value.integer },
                    .float => Value{ .f64 = value.float },
                    .bool => Value{ .bool = value.bool },

                    // .array => {
                    //     // Optional: handle arrays if needed
                    //     Value{ .str = "<array>" };
                    // },

                    else => {
                        std.debug.print("⚠️ Skipping unsupported metadata value type for key {s}\n", .{key});
                        continue;
                    },
                };

                try meta_map.put(key, val);
            }
            continue;
        }
        //std.debug.print("Tensor: {s}\n", .{name});

        if (tensor_obj != .object) continue;
        const fields = tensor_obj.object;

        const dtype = fields.get("dtype").?.string;
        const shape_val = fields.get("shape").?.array;
        const offsets_val = fields.get("data_offsets").?.array;

        // if (dtype_val != .string or shape_val != .array or offsets_val != .array)
        //     return error.InvalidField;
        std.debug.print("  dtype: {s}\n", .{dtype});

        std.debug.print("  shape:{any} ", .{shape_val.items});
        std.debug.print("offsets_val:{any} ", .{offsets_val.items});
        var shape = try allocator.alloc(u64, shape_val.items.len);
        for (0.., shape_val.items) |i, dim| {
            if (dim != .integer) return error.InvalidShape;
            std.debug.print("{d} ", .{dim.integer});
            shape[i] = @as(u64, @intCast(dim.integer));
        }

        if (offsets_val.items.len != 2) return error.InvalidOffsets;
        const start_offset = @as(u64, @intCast(offsets_val.items[0].integer));
        const end_offset = @as(u64, @intCast(offsets_val.items[1].integer));
        if (end_offset <= start_offset) {
            std.debug.print("❌ Invalid offset for tensor: {s} (start={}, end={})\n", .{ name, start_offset, end_offset });
            return error.InvalidOffset;
        }
        std.debug.print("Raw data start of {s}: {any}\n", .{ name, buffer[start_offset..@min(start_offset + 32, end_offset)] });

        // std.debug.print("  data: start={} end={} (len={})\n", .{
        //     start_offset, end_offset, end_offset - start_offset,
        // });
        const element_count = shape_len_product(shape);
        const elem = TensorInfo{ .dtype = dtype, .shape = shape, .data_offsets = .{ .start = 0, .end = 0 } };
        const bytes_per_elem = elem.bitsize() / 8;
        const expected_size = element_count * bytes_per_elem;
        const actual_size = end_offset - start_offset;

        if (expected_size != actual_size) {
            std.debug.print("❌ Tensor {s} expected {d} bytes but got {d} bytes\n", .{ name, expected_size, actual_size });
        }
        if (end_offset > buffer.len or start_offset > end_offset) return error.InvalidOffset;
        const tensor = TensorInfo{
            .shape = shape,
            .dtype = dtype,
            .data_offsets = .{
                .end = end_offset,
                .start = start_offset,
            },
        };
        try tensor_entries.append(TensorEntry{ .name = name, .info = tensor });
        //const tensor_data = buffer[start_offset..end_offset];
        //std.debug.print("  first byte: {d}\n", .{tensor_data[0]});
    }
    const metadata = try Metadata.init(allocator, model_name, meta_map, tensor_entries.items);

    std.debug.print("Safetensors Parsing complete.\n", .{});
    return metadata;
}

pub fn read(model_name: []const u8, allocator: std.mem.Allocator) !void {
    const model = try registry.findModelErrorless(model_name);
    const found_model = model.?;

    std.debug.print("Loading model: {s}\n", .{found_model.name});

    const buffer = try found_model.loadSafetensorsBuffer(allocator);
    defer allocator.free(buffer);
    _ = try parseSafetensorsFromBuffer(allocator, model_name, buffer);
}

fn shape_len_product(shape: []u64) usize {
    var n: usize = 1;
    for (shape) |d| {
        n *= @as(usize, @intCast(d));
    }
    return n;
}
