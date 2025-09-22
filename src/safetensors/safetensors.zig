const std = @import("std");
const registry = @import("../registry/model_registry.zig");
const FileReader = @import("../ggml/reader.zig").FileReader;
const Value = @import("../ggml/KV.zig").Value;
const Metadata = @import("./metadata.zig").Metadata;
const TensorInfo = @import("./tensor_info.zig").TensorInfo;
const TensorEntry = @import("./types.zig").TensorEntry;
const OffsetEntry = @import("./types.zig").OffsetEntry;
const utils = @import("./utils.zig");

pub const SafeTensors = struct {
    metadata: Metadata,
    data: []u8,
};

// TODO: comparing from python safetensors output it seems that offsets in the json are relative to the start of the file, not after the header
// e.g Raw data start of model.embed_tokens.weight: { 128, 153, 0, 0, 0, 0 ...}
// Python: [116, 187, 118, 189, 139, 59,
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
    var iter = root.object.iterator();

    var tensor_entries = std.ArrayList(TensorEntry).init(allocator);

    while (iter.next()) |kv| {
        const name = kv.key_ptr.*;
        const tensor_obj = kv.value_ptr.*;
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
        const start_offset_rel = @as(u64, @intCast(offsets_val.items[0].integer));
        const end_offset_rel = @as(u64, @intCast(offsets_val.items[1].integer));

        // NOTE: seems this was the whole issue about the offesets being wrong
        const metadata_bytesize = 8 + length; // 8 for header size prefix + JSON header size

        const start_offset = metadata_bytesize + start_offset_rel;
        const end_offset = metadata_bytesize + end_offset_rel;
        if (end_offset <= start_offset) {
            std.debug.print("❌ Invalid offset for tensor: {s} (start={}, end={})\n", .{ name, start_offset, end_offset });
            return error.InvalidOffset;
        }
        std.debug.print("Raw data start of {s}: {any}\n", .{ name, buffer[start_offset..@min(start_offset + 32, end_offset)] });
        const tensor_bytes = buffer[start_offset..end_offset];

        if (std.mem.endsWith(u8, name, "norm.weight")) {
            const dims = [_]usize{ 1152, 1, 1, 1 }; // example
            //const rows = shape[0];
            //const cols = shape[1];
            //const dims = [_]usize{ @as(usize, @intCast(rows)), @as(usize, @intCast(cols)) };
            std.debug.print("  Special tensor (LayerNorm weight) dims: {any}\n", .{dims});

            try dumpTensor(tensor_bytes, &dims, dtype);
        }

        // std.debug.print("  data: start={} end={} (len={})\n", .{
        //     start_offset, end_offset, end_offset - start_offset,
        // });
        const element_count = utils.shape_len_product(shape);
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

pub fn dumpTensor(tensor_bytes: []const u8, dims: []const usize, dtype: []const u8) !void {
    var stdout = std.io.getStdOut().writer();

    const total_elems = blk: {
        var prod: usize = 1;
        for (dims) |d| prod *= d;
        break :blk prod;
    };

    if (std.mem.eql(u8, dtype, "F32")) {
        for (0..total_elems) |i| {
            const start = i * 4;

            const ptr: *const [4]u8 = @as(*const [4]u8, @ptrCast(&tensor_bytes[start]));
            const bits = std.mem.readInt(u32, ptr, .little);
            const val = @as(f32, @bitCast(bits));
            try stdout.print("{d:.6} ", .{val});
            if ((i + 1) % dims[dims.len - 1] == 0) {
                try stdout.print("\n", .{});
            }
        }
    } else if (std.mem.eql(u8, dtype, "BF16")) {
        for (0..8) |i| {
            const start = i * 2;
            const ptr: *const [2]u8 = @as(*const [2]u8, @ptrCast(&tensor_bytes[start]));
            const bits = std.mem.readInt(u16, ptr, .little);
            const val = utils.bf16ToF32(bits);
            try stdout.print("{d:.6} ", .{val});
            if ((i + 1) % dims[dims.len - 1] == 0) {
                try stdout.print("\n", .{});
            }
        }
    } else {
        try stdout.print("Unsupported dtype: {s}\n", .{dtype});
    }
}
