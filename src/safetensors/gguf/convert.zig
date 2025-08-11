const std = @import("std");
const GGUFWriter = @import("../../ggml/writer.zig").GGUFWriter;
const Value = @import("../../ggml/KV.zig").Value;
const registry = @import("../../registry/model_registry.zig");
const TensorNameMap = @import("../../ggml/tensor_map.zig").TensorNameMap;
const ModelArch = @import("../../ggml/constants.zig").ModelArch;
const Metadata = @import("../tensor.zig").Metadata;
const TensorInfo = @import("../tensor.zig").TensorInfo;
const utils = @import("utils.zig");
const quant = @import("quant.zig");
const ParseTensorType = @import("../../ggml/types.zig").ParseTensorType;
const TypeSize = @import("../../ggml/types.zig").TypeSize;
const mapDtypeToGGML = @import("../types.zig").mapDtypeToGGML;
const parseSafetensorsFromBuffer = @import("../tensor.zig").parseSafetensorsFromBuffer;
const parseSafetensorsFromBufferV2 = @import("../tensor.zig").parseSafetensorsFromBufferV2;
const ggufPadding = @import("../../ggml/gguf.zig").ggufPadding;
const getTensorNameMap = @import("../../ggml/tensor_map.zig").getTensorNameMap;
const writeSentencePieceTokenizerVocab = @import("general.zig").writeSentencePieceTokenizerVocab;
const writeGeneralMetadata = @import("general.zig").writeGeneralMetadata;

fn writeGGUFHeader(writer: *GGUFWriter, metadata: *Metadata) !void {
    const version: u32 = 3;
    try writer.writer.writeAll("GGUF");
    writer.advance(4);
    try writer.writeU32(version);
    try writer.writeU64(@as(u64, metadata.tensors.items.len));
    // TODO fix dynamic counts
    const general_kv_count: u64 = 6;
    const tokenizer_kv_count: u64 = 13;
    const metadata_kv_count = @as(u64, metadata.metadata.count());
    const total_kv_count = general_kv_count + tokenizer_kv_count + metadata_kv_count;
    try writer.writeU64(total_kv_count);
}

fn writeExtraMetadataKV(writer: *GGUFWriter, metadata: *Metadata) !void {
    var kv_iter = metadata.metadata.iterator();
    while (kv_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        std.debug.print("key {s}\n", .{key});
        if (std.mem.startsWith(u8, key, "format")) continue;
        if (std.mem.startsWith(u8, key, "tensor.") or metadata.index_map.contains(key)) continue;

        const val = entry.value_ptr.*;
        try utils.writeKeyValue(writer, key, val);
    }
}

pub fn prepare_tensorsV2(
    allocator: std.mem.Allocator,
    metadata: *Metadata,
    safetensors_buffer: []const u8,
    writer: *GGUFWriter,
    out_file: *std.fs.File,
) !void {
    const alignment: u64 = 32;

    const block_count_val = metadata.get(allocator, "block_count") orelse {
        return error.MissingBlockCount;
    };
    const block_count: u32 = block_count_val.u32;

    var tensor_map = try getTensorNameMap(allocator, ModelArch.GEMMA3, block_count);
    const offkeys = try metadata.offsetKeys(allocator);

    const OffsetPatch = struct {
        tensor_index: usize,
        pos_in_file: usize,
        tensor_info: *const TensorInfo,
        name: []const u8,
    };
    var offset_patch_list = std.ArrayList(OffsetPatch).init(allocator);
    defer offset_patch_list.deinit();

    // 1) Write tensor headers (swap dims for .weight but not _norm.weight), write dtype header (F16 default, F32 for norm)
    for (offkeys, 0..) |entry, index| {
        const name = utils.truncate_name(entry);
        const new_name = try tensor_map.get_name(allocator, name, &[_][]const u8{ ".weight", ".bias" });
        const tensor = &metadata.tensors.items[index];

        try writer.writeString(new_name.?);

        var shape_to_write = tensor.shape;
        if (std.mem.endsWith(u8, new_name.?, ".weight") and !std.mem.endsWith(u8, new_name.?, "_norm.weight")) {
            if (shape_to_write.len >= 2) {
                const tmp = shape_to_write[0];
                shape_to_write[0] = shape_to_write[1];
                shape_to_write[1] = tmp;
            }
        }

        const len = @as(u32, @intCast(shape_to_write.len));
        try writer.writeU32(len);
        for (shape_to_write) |dim| {
            try writer.writeU64(dim);
        }

        var out_dtype: []const u8 = "F16";
        if (std.mem.endsWith(u8, new_name.?, "_norm.weight")) {
            out_dtype = "F32"; // override header dtype for norm weights
        }

        const kind = @as(u32, @intCast(@intFromEnum(try mapDtypeToGGML(out_dtype))));
        try writer.writeU32(kind);

        const offset_placeholder_pos = writer.position;
        try writer.writeU64(0);

        try offset_patch_list.append(.{
            .tensor_index = index,
            .pos_in_file = offset_placeholder_pos,
            .tensor_info = tensor,
            .name = new_name.?,
        });
    }

    // 2) Align and record data base
    try writer.writePadding(alignment);
    const data_base = writer.position;

    // 3) Write tensor data and patch offsets
    for (offset_patch_list.items) |patch| {
        const tensor = patch.tensor_info;
        const actual_offset = @as(u64, @intCast(writer.position - data_base));

        // patch header offset (absolute)
        const current_pos = writer.position;
        try out_file.seekTo(@intCast(patch.pos_in_file));
        try out_file.writeAll(&std.mem.toBytes(actual_offset));
        try out_file.seekTo(@intCast(current_pos));

        const start = @as(usize, @intCast(tensor.data_offsets.start));
        const end = @as(usize, @intCast(tensor.data_offsets.end));
        const tensor_data = safetensors_buffer[start..end];

        const is_norm = std.mem.endsWith(u8, patch.name, "_norm.weight");

        if (is_norm) {
            // We need to write F32 values (header says F32 for norms). Convert from original dtype -> F32,
            // add 1.0 to each element, then write F32 bytes.
            const src_dtype = tensor.dtype; // original safetensors dtype string, e.g. "F16", "F32", "BF16"
            if (std.mem.eql(u8, src_dtype, "F32")) {
                // src is f32: read 4 bytes per element, add 1.0
                const count = @as(usize, @intCast(tensor_data.len / 4));
                var buf = try allocator.alloc(f32, count);
                defer allocator.free(buf);

                var i: usize = 0;
                while (i < count) : (i += 1) {
                    const b0 = @as(u32, tensor_data[i * 4 + 0]);
                    const b1 = @as(u32, tensor_data[i * 4 + 1]);
                    const b2 = @as(u32, tensor_data[i * 4 + 2]);
                    const b3 = @as(u32, tensor_data[i * 4 + 3]);
                    const bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    buf[i] = @as(f32, @bitCast(bits)) + 1.0;
                }

                // write buf as bytes
                const bytes_to_write = @as(usize, @intCast(count * @sizeOf(f32)));
                try writer.writer.writeAll(@as([*]const u8, @ptrCast(buf.ptr))[0..bytes_to_write]);
                writer.advance(bytes_to_write);
            } else if (std.mem.eql(u8, src_dtype, "F16")) {
                // src is f16: each element is 2 bytes. Convert half -> f32, add 1.0
                const count = @as(usize, @intCast(tensor_data.len / 2));
                var buf = try allocator.alloc(f32, count);
                defer allocator.free(buf);

                var i: usize = 0;
                while (i < count) : (i += 1) {
                    const lo = @as(u16, tensor_data[i * 2 + 0]);
                    const hi = @as(u16, tensor_data[i * 2 + 1]);
                    const halfBits = lo | (@as(u16, hi) << 8);
                    buf[i] = quant.halfToF32(halfBits) + 1.0;
                }

                const bytes_to_write = @as(usize, @intCast(count * @sizeOf(f32)));
                try writer.writer.writeAll(@as([*]const u8, @ptrCast(buf.ptr))[0..bytes_to_write]);
                writer.advance(bytes_to_write);
            } else if (std.mem.eql(u8, src_dtype, "BF16")) {
                // src is bfloat16: 2 bytes per element, top 16 bits of f32.
                const count = @as(usize, @intCast(tensor_data.len / 2));
                var buf = try allocator.alloc(f32, count);
                defer allocator.free(buf);

                var i: usize = 0;
                while (i < count) : (i += 1) {
                    const lo = @as(u32, tensor_data[i * 2 + 0]);
                    const hi = @as(u32, tensor_data[i * 2 + 1]);
                    const halfBits = lo | (hi << 8); // 16-bit bfloat
                    const bits = halfBits << 16; // to f32 top 16 bits
                    buf[i] = @as(f32, @bitCast(bits)) + 1.0;
                }

                const bytes_to_write = @as(usize, @intCast(count * @sizeOf(f32)));
                try writer.writer.writeAll(@as([*]const u8, @ptrCast(buf.ptr))[0..bytes_to_write]);
                writer.advance(bytes_to_write);
            } else {
                // Unknown dtype - fallback: write raw bytes (but this will likely break offsets)
                try writer.writer.writeAll(tensor_data);
                writer.advance(tensor_data.len);
            }
        } else {
            // Non-norm tensors: write raw bytes as-is (no +1)
            try writer.writer.writeAll(tensor_data);
            writer.advance(tensor_data.len);
        }

        // pad to alignment based on writer.position
        try writer.writePadding(alignment);
    }
}

fn prepare_tensors(
    allocator: std.mem.Allocator,
    metadata: *Metadata,
    safetensors_buffer: []const u8,
    writer: *GGUFWriter,
) !void {
    // ✅ Tensor headers
    const alignment = 32;
    const tensor_count = @as(u64, metadata.tensors.items.len);
    var tensor_data_offsets = try allocator.alloc(u64, tensor_count);
    defer allocator.free(tensor_data_offsets);

    var offset: u64 = 0;
    for (metadata.tensors.items, 0..) |tensor, i| {
        tensor_data_offsets[i] = offset;
        const tensor_type = try ParseTensorType(tensor.dtype);
        var element_count: usize = 1;
        for (tensor.shape) |dim| {
            element_count *= dim;
        }
        const size = @as(u64, @intCast(element_count)) * TypeSize(tensor_type);
        offset += size;
        const pad = ggufPadding(offset, alignment);
        offset += pad;
    }
    var block_count: u32 = 0;
    const block_count_val = metadata.get(allocator, "block_count") orelse {
        return error.MissingBlockCount;
    };
    block_count = block_count_val.u32;
    var tensor_map = try getTensorNameMap(allocator, ModelArch.GEMMA3, block_count);

    const offkeys = try metadata.offsetKeys(allocator);
    for (offkeys, 0..) |entry, index| {
        const name = utils.truncate_name(entry);
        const new_name = try tensor_map.get_name(allocator, name, &[_][]const u8{ ".weight", ".bias" });

        const tensor = &metadata.tensors.items[index];

        try writer.writeString(new_name.?);

        const len = @as(u32, @intCast(tensor.shape.len));
        try writer.writeU32(len);

        for (tensor.shape) |dim| {
            try writer.writeU64(dim);
        }
        if (std.mem.endsWith(u8, new_name.?, "_norm.weight")) {
            tensor.dtype = "F32";
        }
        const kind = @as(u32, @intCast(@intFromEnum(try mapDtypeToGGML(tensor.dtype))));
        try writer.writeU32(kind);
        try writer.writeU64(tensor_data_offsets[index]);

        //std.debug.print("Tensor {s} at offset {d}\n", .{ name, tensor_data_offsets[index] });
    }

    // ✅ Padding
    const header_size = offset;
    const data_pad = ggufPadding(header_size, alignment);
    if (data_pad > 0) {
        try writer.writePadding(alignment);
    }
    // ✅ Tensor data
    for (metadata.tensors.items) |tensor| {
        const start = tensor.data_offsets.start;
        const end = tensor.data_offsets.end;
        const tensor_data = safetensors_buffer[start..end];
        try writer.writer.writeAll(tensor_data);
        writer.advance(tensor_data.len);

        const pad = ggufPadding(tensor_data.len, alignment);
        if (pad > 0) {
            try writer.writePadding(alignment);
        }
    }
}

pub fn convertToGGUFFromSafeTensors(
    allocator: std.mem.Allocator,
    metadata: *Metadata,
    safetensors_buffer: []const u8,
    file_writer: std.io.AnyWriter,
    basename: []const u8,
    architecture: []const u8,
    model_name: []const u8,
    tokenizer_path: []const u8,
    quant_version: u32,
    out_file: *std.fs.File,
) !void {
    var writer = GGUFWriter.init(file_writer);

    try writeGGUFHeader(&writer, metadata);

    try writeGeneralMetadata(
        &writer,
        basename,
        architecture,
        model_name,
    );

    try writeExtraMetadataKV(&writer, metadata);

    try writeSentencePieceTokenizerVocab(
        allocator,
        &writer,
        tokenizer_path,
    );

    try writer.writeString("general.quantization_version");
    try writer.writeU32(4); // ggufTypeUint32
    try writer.writeU32(quant_version);

    try prepare_tensorsV2(
        allocator,
        metadata,
        safetensors_buffer,
        &writer,
        out_file,
    );
}

fn prepare_metadata(allocator: std.mem.Allocator, metadata: *Metadata, model: registry.ModelInfo, model_files: []const []const u8) !void {
    if (utils.indexOfStringInList(model_files, "config.json")) |i| {
        if (try utils.tryLoadJson(allocator, model, model_files[i])) |cfg| {
            const obj = cfg.object;

            if (obj.get("max_position_embeddings")) |v| {
                try metadata.put(
                    allocator,
                    "context_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            } else {
                try metadata.put(
                    allocator,
                    "context_length",
                    .{ .u32 = 131072 },
                );
            }

            if (obj.get("hidden_size")) |v| {
                try metadata.put(
                    allocator,
                    "embedding_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            }
            if (obj.get("num_hidden_layers")) |v| {
                const b_count = @as(u32, (@intCast(v.integer)));
                try metadata.put(
                    allocator,
                    "block_count",
                    .{ .u32 = b_count },
                );
            }
            if (obj.get("intermediate_size")) |v| {
                try metadata.put(
                    allocator,
                    "feed_forward_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            }

            if (obj.get("num_attention_heads")) |v| {
                try metadata.put(
                    allocator,
                    "attention.head_count",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            } else {
                try metadata.put(
                    allocator,
                    "attention.head_count",
                    .{ .u32 = 8 },
                );
            }

            if (obj.get("rms_norm_eps")) |v| {
                try metadata.put(
                    allocator,
                    "attention.layer_norm_rms_epsilon",
                    .{ .f32 = @as(f32, (@floatCast(v.float))) },
                );
            } else {
                try metadata.put(
                    allocator,
                    "attention.layer_norm_rms_epsilon",
                    .{ .f32 = (@as(f32, (@floatCast(0.000001)))) },
                );
            }
            if (obj.get("head_dim")) |v| {
                try metadata.put(
                    allocator,
                    "attention.key_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                try metadata.put(
                    allocator,
                    "attention.value_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            }
            try metadata.metadata.put(
                "general.file_type",
                .{ .u32 = (@as(u32, (@intCast(1)))) },
            );

            if (obj.get("rope_theta")) |v| {
                const freq_base_val = switch (v) {
                    .float => @as(f32, @floatCast(v.float)),
                    .integer => @as(f32, @floatFromInt(v.integer)),
                    else => return error.UnexpectedValueType,
                };
                try metadata.put(
                    allocator,
                    "rope.freq_base",
                    .{ .f32 = freq_base_val },
                );
            } else {
                try metadata.put(
                    allocator,
                    "rope.freq_base",
                    .{ .f32 = 1000000.0 },
                );
            }

            if (obj.get("sliding_window")) |v| {
                try metadata.put(
                    allocator,
                    "attention.sliding_window",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            }
            if (obj.get("num_key_value_heads")) |v| {
                try metadata.put(
                    allocator,
                    "attention.head_count_kv",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
            }

            if (obj.get("rope_scaling")) |v| {
                if (v == .object) {
                    if (v.object.get("rope_type")) |rt| {
                        if (rt == .string and std.mem.eql(u8, rt.string, "linear")) {
                            try metadata.put(
                                allocator,
                                "rope.scaling.type",
                                .{ .str = "linear" },
                            );
                            if (v.object.get("factor")) |f| {
                                const factor_val = switch (f) {
                                    .float => f.float,
                                    .integer => @as(f64, @floatFromInt(f.integer)),
                                    else => return error.UnexpectedValueType,
                                };
                                try metadata.put(
                                    allocator,
                                    "rope.scaling.factor",
                                    .{ .f64 = factor_val },
                                );
                            }
                        }
                    }
                }
            }

            if (obj.get("attn_logit_softcapping")) |v| {
                if (v != .null) return error.UnsupportedFieldPresent;
            }
            if (obj.get("final_logit_softcapping")) |v| {
                if (v != .null) return error.UnsupportedFieldPresent;
            }
        }
    }
}

pub fn convert(model_name: []const u8, output_path: []const u8, allocator: std.mem.Allocator) !void {
    const fs = std.fs;

    // Open and read the safetensors input file
    const model = try registry.findModelErrorless(model_name);
    const found_model = model.?;
    const model_files = found_model.files;

    std.debug.print("Loading model: {s}\n", .{found_model.name});

    const buffer = try found_model.loadSafetensorsBuffer(allocator);
    defer allocator.free(buffer);

    // Parse safetensors metadata
    var metadata = try parseSafetensorsFromBuffer(allocator, model_name, buffer);
    try prepare_metadata(allocator, &metadata, found_model, model_files);

    var output_file = try fs.cwd().createFile(output_path, .{ .read = false, .truncate = true });
    defer output_file.close();

    const writer = output_file.writer().any();
    // Convert to GGUF and write to file
    const tokenizer_path = try found_model.localFilePath(found_model.name, "tokenizer_export.json");
    try convertToGGUFFromSafeTensors(
        allocator,
        &metadata,
        buffer,
        writer,
        found_model.name,
        found_model.name,
        found_model.name,
        tokenizer_path,
        2,
        &output_file,
    );

    std.debug.print("✓ Converted '{s}' → '{s}'\n with metadata count: {d}", .{ model_name, output_path, metadata.metadata.count() });
}
