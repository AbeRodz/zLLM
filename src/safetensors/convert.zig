const std = @import("std");
const GGUFWriter = @import("../ggml/writer.zig").GGUFWriter;
const Value = @import("../ggml/KV.zig").Value;
const parseTokenizerJson = @import("../ggml/tokenizer.zig").parseTokenizerJson;
const GGUFDataType = @import("../ggml/types.zig").GGUFDataType;
const mapDtypeToGGML = @import("types.zig").mapDtypeToGGML;
const registry = @import("../registry/model_registry.zig");
const parseSafetensorsFromBuffer = @import("tensor.zig").parseSafetensorsFromBuffer;
const ggufPadding = @import("../ggml/gguf.zig").ggufPadding;
const Metadata = @import("tensor.zig").Metadata;

fn writeGGUFString(writer: *GGUFWriter, str: []const u8) !void {
    try writer.writeU64(@intCast(str.len));
    try writer.writer.writeAll(str);
    writer.advance(str.len);
}

pub fn modelWeightCountRoundedNotation(
    comptime min_digits: usize,
    model_params_count: u64,
    buf: []u8,
) ![]const u8 {
    var scaled_model_params: f64 = 0;
    var scale_suffix: []const u8 = "";

    if (model_params_count > 1_000_000_000_000) {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-12;
        scale_suffix = "T";
    } else if (model_params_count > 1_000_000_000) {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-9;
        scale_suffix = "B";
    } else if (model_params_count > 1_000_000) {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-6;
        scale_suffix = "M";
    } else {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-3;
        scale_suffix = "K";
    }

    const rounded = @round(scaled_model_params);

    var temp: [32]u8 = undefined;
    const rounded_str = try std.fmt.bufPrint(&temp, "{d}", .{rounded});
    const trimmed = std.mem.trimLeft(u8, rounded_str, "0");

    const fix: usize = if (trimmed.len >= min_digits) 0 else min_digits - trimmed.len;

    return std.fmt.bufPrint(buf, "{any}{d}{s}", .{ scaled_model_params, fix, scale_suffix });
}

pub fn sizeLabel(
    comptime min_digits: usize,
    total_params: u64,
    shared_params: i64,
    expert_params: i64,
    expert_count: u32,
    buf: []u8,
) ![]const u8 {
    if (expert_count > 0) {
        const combined = @abs(shared_params) + @abs(expert_params);
        var temp_buf: [64]u8 = undefined;
        const pretty = try modelWeightCountRoundedNotation(min_digits, combined, &temp_buf);
        return std.fmt.bufPrint(buf, "{d}x{s}", .{ expert_count, pretty });
    } else {
        return modelWeightCountRoundedNotation(min_digits, total_params, buf);
    }
}

pub fn convertToGGUFFromSafeTensors(
    allocator: std.mem.Allocator,
    metadata: Metadata,
    safetensors_buffer: []const u8,
    file_writer: std.io.AnyWriter,
    basename: []const u8,
    architecture: []const u8,
    model_name: []const u8,
    tokenizer_path: []const u8,
    quant_version: u32,
) !void {
    var writer = GGUFWriter.init(file_writer);

    const version: u32 = 3;

    // Write GGUF header
    try writer.writer.writeAll("GGUF");
    writer.advance(4);
    try writer.writeU32(version);

    const tensor_count = @as(u64, metadata.tensors.items.len);
    try writer.writeU64(tensor_count);

    const general_kv_count: u64 = 7;
    const tokenizer_kv_count: u64 = 13;
    const metadata_kv_count = @as(u64, metadata.metadata.count());
    const total_kv_count = general_kv_count + tokenizer_kv_count + metadata_kv_count;
    try writer.writeU64(total_kv_count);

    try writeGeneralMetadata(&writer, basename, architecture, model_name, quant_version);
    try writeSentencePieceTokenizerVocab(allocator, &writer, tokenizer_path);
    // Write metadata key-value pairs
    var kv_iter = metadata.metadata.iterator();
    while (kv_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        std.debug.print("{s}\n", .{key});
        if (std.mem.startsWith(u8, key, "format")) {
            continue;
        }
        if (std.mem.startsWith(u8, key, "tensor.") or metadata.index_map.contains(key)) {
            std.debug.print("⚠️ Skipping tensor-like KV key: {s}\n", .{key});
            continue;
        }
        const val = entry.value_ptr.*;

        try writer.writeString(key);

        var type_id: GGUFDataType = undefined;
        var is_array = false;

        switch (val) {
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
            const length: u64 = switch (val) {
                .u8s => val.u8s.len,
                .i8s => val.i8s.len,
                .u16s => val.u16s.len,
                .i16s => val.i16s.len,
                .u32s => val.u32s.len,
                .i32s => val.i32s.len,
                .u64s => val.u64s.len,
                .i64s => val.i64s.len,
                .f32s => val.f32s.len,
                .f64s => val.f64s.len,
                .bools => val.bools.len,
                .strs => val.strs.len,
                else => unreachable,
            };
            try writer.writeU64(length);

            switch (val) {
                .u8s => for (val.u8s) |v| try writer.writeU8(v),
                .i8s => for (val.i8s) |v| try writer.writeI8(v),
                .u16s => for (val.u16s) |v| try writer.writeU16(v),
                .i16s => for (val.i16s) |v| try writer.writeI16(v),
                .u32s => for (val.u32s) |v| try writer.writeU32(v),
                .i32s => for (val.i32s) |v| try writer.writeI32(v),
                .u64s => for (val.u64s) |v| try writer.writeU64(v),
                .i64s => for (val.i64s) |v| try writer.writeI64(v),
                .f32s => for (val.f32s) |v| try writer.writeF32(v),
                .f64s => for (val.f64s) |v| try writer.writeF64(v),
                .bools => for (val.bools) |v| try writer.writeBool(v),
                .strs => for (val.strs) |v| try writer.writeString(v),
                else => unreachable,
            }
        } else {
            switch (val) {
                .u8 => try writer.writeU8(val.u8),
                .i8 => try writer.writeI8(val.i8),
                .u16 => try writer.writeU16(val.u16),
                .i16 => try writer.writeI16(val.i16),
                .u32 => try writer.writeU32(val.u32),
                .i32 => try writer.writeI32(val.i32),
                .u64 => try writer.writeU64(val.u64),
                .i64 => try writer.writeI64(val.i64),
                .f32 => try writer.writeF32(val.f32),
                .f64 => try writer.writeF64(val.f64),
                .bool => try writer.writeBool(val.bool),
                .str => try writer.writeString(val.str),
                else => unreachable,
            }
        }
    }

    // ✅ Tensor headers
    const alignment = 32;
    var tensor_data_offsets = try allocator.alloc(u64, tensor_count);
    defer allocator.free(tensor_data_offsets);

    var offset: u64 = 0;
    for (metadata.tensors.items, 0..) |tensor, i| {
        tensor_data_offsets[i] = offset;
        const size = tensor.data_offsets.end - tensor.data_offsets.start;
        offset += size;
        offset += ggufPadding(offset, alignment);
    }

    var it = metadata.index_map.iterator();
    while (it.next()) |entry| {
        // TODO map safetensors name to GGUF tensor names
        const name = entry.key_ptr.*;
        const index = entry.value_ptr.*;
        const tensor = metadata.tensors.items[index];

        try writer.writeString(name);

        const len = @as(u32, @intCast(tensor.shape.len));
        try writer.writeU32(len);

        for (tensor.shape) |dim| {
            try writer.writeU64(dim);
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
    var metait = metadata.index_map.iterator();
    while (metait.next()) |entry| {
        const index = entry.value_ptr.*;
        const tensor = metadata.tensors.items[index];
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
const GeneralTag = union(enum) {
    str: []const u8,
    u32: u32,
};

pub fn writeSentencePieceTokenizerVocab(allocator: std.mem.Allocator, writer: *GGUFWriter, tokenizer_path: []const u8) !void {
    const tokens = try parseTokenizerJson(allocator, tokenizer_path);
    defer allocator.free(tokens.tokens);
    const count = tokens.tokens.len;
    var pieces = try allocator.alloc([]const u8, count);
    var scores = try allocator.alloc(f32, count);
    var types = try allocator.alloc(u32, count);
    defer allocator.free(pieces);
    defer allocator.free(scores);
    defer allocator.free(types);

    for (tokens.tokens, 0..) |token, i| {
        pieces[i] = token.piece;
        scores[i] = token.score;
        types[i] = token.type;
    }
    // Define the keys and values
    const tokenizer_tags = .{
        .{ "tokenizer.ggml.model", GeneralTag{ .str = "llama" } },
        .{ "tokenizer.ggml.pre", GeneralTag{ .str = "default" } },
    };

    inline for (tokenizer_tags) |tag| {
        const key = tag[0];
        const val = tag[1];

        try writer.writeString(key);
        try writer.writeU32(8); // ggufTypeString
        try writer.writeString(val.str);
    }
    try writer.writeString("tokenizer.ggml.add_space_prefix");
    try writer.writeU32(7); // ggufTypeArray
    try writer.writeBool(false); // ggufTypeArray
    // ✅ Write tokenizer tokens
    try writer.writeString("tokenizer.ggml.tokens");
    try writer.writeU32(9); // ggufTypeArray
    try writer.writeU32(8); // ggufTypeString inside array
    try writer.writeU64(@as(u64, @intCast(count)));
    for (pieces) |piece| {
        try writer.writeString(piece);
    }

    // ✅ Write tokenizer scores
    try writer.writeString("tokenizer.ggml.scores");
    try writer.writeU32(9); // ggufTypeArray
    try writer.writeU32(6); // ggufTypeFloat32 inside array
    try writer.writeU64(@as(u64, @intCast(count)));
    for (scores) |score| {
        try writer.writeF32(score);
    }

    // ✅ Write tokenizer token types
    try writer.writeString("tokenizer.ggml.token_type");
    try writer.writeU32(9); // ggufTypeArray
    try writer.writeU32(5); // ggufTypeInt32 inside array
    try writer.writeU64(@as(u64, @intCast(count)));
    for (types) |typ| {
        try writer.writeU32(typ);
    }

    var it = tokens.special_tokens.iterator();
    while (it.next()) |entry| {
        const typ = entry.key_ptr.*;
        const id = entry.value_ptr.*;
        const mapped_typ = if (std.mem.eql(u8, typ, "pad"))
            "padding"
        else if (std.mem.eql(u8, typ, "unk"))
            "unknown"
        else
            typ;
        const key = try std.fmt.allocPrint(allocator, "tokenizer.ggml.{s}_token_id", .{mapped_typ});
        defer allocator.free(key);

        try writer.writeString(key);
        try writer.writeU32(4); // ggufTypeUInt32
        try writer.writeU32(@as(u32, @intCast(id)));
    }
    var add_st_it = tokens.add_special_tokens.iterator();
    while (add_st_it.next()) |entry| {
        const typ = entry.key_ptr.*;
        const val = entry.value_ptr.*;

        const key = try std.fmt.allocPrint(allocator, "tokenizer.ggml.add_{s}_token", .{typ});
        std.debug.print("Tokenizer special token {s} tokens\n", .{key});
        defer allocator.free(key);

        try writer.writeString(key);
        try writer.writeU32(7); // ggufTypeBool
        try writer.writeBool(val);
    }
    try writer.writeString("tokenizer.chat_template");
    try writer.writeU32(8); // ggufTypeString
    try writer.writeString(tokens.chat_template);

    std.debug.print("Tokenizer vocab set with {d} tokens\n", .{pieces.len});
}

pub fn writeGeneralMetadata(
    writer: *GGUFWriter,
    basename: []const u8,
    architecture: []const u8,
    model_name: []const u8,
    quant_version: u32,
) !void {
    var output: [64]u8 = undefined;
    // TODO Get real parameters count dynamically
    const label = try sizeLabel(2, 1_000_000_000, 0, 0, 0, &output);
    std.debug.print("Label: {s}\n", .{label});
    const general_tags = .{
        .{ "general.architecture", GeneralTag{ .str = architecture } },
        .{ "general.type", GeneralTag{ .str = "model" } },
        .{ "general.name", GeneralTag{ .str = model_name } },
        .{ "general.basename", GeneralTag{ .str = basename } },
        .{ "general.file_type", GeneralTag{ .u32 = 1 } },
        .{ "general.quantization_version", GeneralTag{ .u32 = quant_version } },
        .{ "general.size_label", GeneralTag{ .str = label } },
    };

    inline for (general_tags) |tag| {
        const key = tag[0];
        const value = tag[1];

        try writer.writeString(key);

        switch (value) {
            .str => {
                try writer.writeU32(8); // ggufTypeString
                try writer.writeString(value.str);
            },
            .u32 => {
                try writer.writeU32(4); // ggufTypeUint32
                try writer.writeU32(value.u32);
            },
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
    var metadata = try parseSafetensorsFromBuffer(allocator, buffer);
    if (indexOfStringInList(model_files, "config.json")) |i| {
        if (try tryLoadJson(allocator, found_model, model_files[i])) |cfg| {
            const obj = cfg.object;

            // Required + optional fields with defaults
            if (obj.get("max_position_embeddings")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "context_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.context_length", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            } else {
                try metadata.put(
                    allocator,
                    model_name,
                    "context_length",
                    .{ .u32 = 131072 },
                );
                //try metadata.metadata.put("llm.context_length", .{ .u32 = 131072 });
            }

            if (obj.get("hidden_size")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "embedding_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.embedding_length", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            }

            if (obj.get("intermediate_size")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "feed_forward_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.feed_forward_length", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            }

            if (obj.get("num_attention_heads")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.head_count",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.attention.head_count", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            } else {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.head_count",
                    .{ .u32 = 8 },
                );
                //try metadata.metadata.put("llm.attention.head_count", .{ .u32 = 8 });
            }

            if (obj.get("rms_norm_eps")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.layer_norm_rms_epsilon",
                    .{ .f32 = @as(f32, (@floatCast(v.float))) },
                );
                //try metadata.metadata.put("llm.attention.layer_norm_rms_epsilon", .{ .f32 = @as(f32, (@floatCast(v.float))) });
            } else {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.layer_norm_rms_epsilon",
                    .{ .f32 = (@as(f32, (@floatCast(0.000001)))) },
                );
                //    try metadata.metadata.put("llm.attention.layer_norm_rms_epsilon", .{ .f32 = (@as(f32, (@floatCast(0.000001)))) });
            }
            if (obj.get("head_dim")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.key_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.value_length",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.attention.key_length", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
                //try metadata.metadata.put("llm.attention.value_length", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            }

            if (obj.get("num_hidden_layers")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "block_count",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.block_count", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            }

            if (obj.get("rope_theta")) |v| {
                const freq_base_val = switch (v) {
                    .float => @as(f32, @floatCast(v.float)),
                    .integer => @as(f32, @floatFromInt(v.integer)),
                    else => return error.UnexpectedValueType,
                };
                try metadata.put(
                    allocator,
                    model_name,
                    "rope.freq_base",
                    .{ .f32 = freq_base_val },
                );
                //try metadata.metadata.put("llm.rope.freq_base", .{ .f32 = freq_base_val });
            } else {
                try metadata.put(
                    allocator,
                    model_name,
                    "rope.freq_base",
                    .{ .f32 = 1000000.0 },
                );
                //try metadata.metadata.put("llm.rope.freq_base", .{ .f32 = 1000000.0 });
            }

            if (obj.get("sliding_window")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.sliding_window",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.attention.sliding_window", .{ .u32 = (@as(u32, (@intCast(v.integer)))) });
            }
            if (obj.get("num_key_value_heads")) |v| {
                try metadata.put(
                    allocator,
                    model_name,
                    "attention.head_count_kv",
                    .{ .u32 = (@as(u32, (@intCast(v.integer)))) },
                );
                //try metadata.metadata.put("llm.attention.head_count_kv", .{ .u32 = @as(u32, (@intCast(v.integer))) });
            }

            // rope_scaling check
            if (obj.get("rope_scaling")) |v| {
                if (v == .object) {
                    if (v.object.get("rope_type")) |rt| {
                        if (rt == .string and std.mem.eql(u8, rt.string, "linear")) {
                            try metadata.put(
                                allocator,
                                model_name,
                                "rope.scaling.type",
                                .{ .str = "linear" },
                            );
                            // try metadata.metadata.put("llm.rope.scaling.type", .{ .str = "linear" });
                            if (v.object.get("factor")) |f| {
                                const factor_val = switch (f) {
                                    .float => f.float,
                                    .integer => @as(f64, @floatFromInt(f.integer)),
                                    else => return error.UnexpectedValueType,
                                };
                                try metadata.put(
                                    allocator,
                                    model_name,
                                    "rope.scaling.factor",
                                    .{ .f64 = factor_val },
                                );
                                //try metadata.metadata.put("llm.rope.scaling.factor", .{ .f64 = factor_val });
                            }
                        }
                    }
                }
            }

            // For clarity, explicitly assert unsupported keys are not present (like Python version)
            if (obj.get("attn_logit_softcapping")) |v| {
                if (v != .null) return error.UnsupportedFieldPresent;
            }
            if (obj.get("final_logit_softcapping")) |v| {
                if (v != .null) return error.UnsupportedFieldPresent;
            }
        }
    }

    const output_file = try fs.cwd().createFile(output_path, .{ .read = false, .truncate = true });
    defer output_file.close();

    const writer = output_file.writer().any();

    // Convert to GGUF and write to file
    const tokenizer_path = try found_model.localFilePath(found_model.name, "tokenizer_export.json");
    try convertToGGUFFromSafeTensors(allocator, metadata, buffer, writer, found_model.name, found_model.name, found_model.name, tokenizer_path, 2);

    std.debug.print("✓ Converted '{s}' → '{s}'\n with metadata count: {d}", .{ model_name, output_path, metadata.metadata.count() });
}

fn indexOfStringInList(haystack: []const []const u8, needle: []const u8) ?usize {
    for (haystack, 0..) |val, i| {
        if (std.mem.eql(u8, val, needle)) return i;
    }
    return null;
}

inline fn tryLoadJson(allocator: std.mem.Allocator, model: registry.ModelInfo, file_path: []const u8) !?std.json.Value {
    const path = try model.localFilePath(model.name, file_path);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const content = try allocator.alloc(u8, file_size);
    _ = try file.readAll(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    return parsed.value;
}
