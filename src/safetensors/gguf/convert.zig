const std = @import("std");

pub const QuantType = enum {
    f16,
    q8_0,
    q4_k,
};

const GGUFWriter = @import("../../ggml/writer.zig").GGUFWriter;
const Value = @import("../../ggml/KV.zig").Value;
const registry = @import("../../registry/model_registry.zig");
const TensorNameMap = @import("../../ggml/tensor_map.zig").TensorNameMap;
const ModelArch = @import("../../ggml/constants.zig").ModelArch;
const ModelArchNames = @import("../../ggml/constants.zig").ModelArchNames;
const Metadata = @import("../metadata.zig").Metadata;
const TensorInfo = @import("../tensor_info.zig").TensorInfo;
const utils = @import("utils.zig");
const quant = @import("quant.zig");
const ParseTensorType = @import("../../ggml/types.zig").ParseTensorType;
const TypeSize = @import("../../ggml/types.zig").TypeSize;
const mapDtypeToGGML = @import("../types.zig").mapDtypeToGGML;
const parseSafetensorsFromBuffer = @import("../safetensors.zig").parseSafetensorsFromBuffer;
const ggufPadding = @import("../../ggml/gguf.zig").ggufPadding;
const getTensorNameMap = @import("../../ggml/tensor_map.zig").getTensorNameMap;
const writeSentencePieceTokenizerVocab = @import("general.zig").writeSentencePieceTokenizerVocab;
const writeBPETokenizerVocab = @import("general.zig").writeBPETokenizerVocab;
const writeGeneralMetadata = @import("general.zig").writeGeneralMetadata;

/// Describes which tokenizer files are present and how many GGUF KV entries
/// the tokenizer writer will produce.  The caller passes this to writeGGUFHeader
/// so the count in the file header is always exact.
const TokenizerSpec = union(enum) {
    /// SentencePiece export JSON (tokenizer_export.json)
    sentencepiece: []const u8,
    /// HuggingFace BPE (tokenizer.json + tokenizer_config.json)
    bpe: struct { json: []const u8, config: []const u8 },

    fn kvCount(self: TokenizerSpec) u64 {
        return switch (self) {
            .sentencepiece => 13,
            .bpe => 9,
        };
    }
};

/// Map a HuggingFace model_type string to the GGUF ModelArch enum.
/// Falls back to LLAMA for unknown types (most modern HF models are LLaMA-style).
fn archFromModelType(model_type: []const u8) ModelArch {
    const map = .{
        .{ "llama",       ModelArch.LLAMA   },
        .{ "mistral",     ModelArch.LLAMA   },
        .{ "gemma",       ModelArch.GEMMA   },
        .{ "gemma2",      ModelArch.GEMMA2  },
        .{ "gemma3",      ModelArch.GEMMA3  },
        .{ "gemma3_text", ModelArch.GEMMA3  },
        .{ "qwen2",       ModelArch.QWEN2   },
        .{ "phi",         ModelArch.PHI2    },
        .{ "phi3",        ModelArch.PHI3    },
        .{ "starcoder2",  ModelArch.STARCODER2 },
        .{ "falcon",      ModelArch.FALCON  },
        .{ "gpt2",        ModelArch.GPT2    },
        .{ "gpt_neox",    ModelArch.GPTNEOX },
    };
    inline for (map) |entry| {
        if (std.mem.eql(u8, model_type, entry[0])) return entry[1];
    }
    return .LLAMA;
}

fn writeGGUFHeader(writer: *GGUFWriter, metadata: *Metadata, tokenizer_kv_count: u64) !void {
    const version: u32 = 3;
    try writer.writer.writeAll("GGUF");
    writer.advance(4);
    try writer.writeU32(version);
    try writer.writeU64(@as(u64, metadata.tensors.items.len));
    const general_kv_count: u64 = 5;
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

/// Reorder Q or K rows in-place for llama.cpp's NeoX RoPE convention.
/// HuggingFace stores each head's rotary pairs as [0..half, half..dim],
/// llama.cpp expects them interleaved as [0, half, 1, half+1, ...].
/// scratch must be at least head_dim*n_cols elements (one head's worth).
fn permuteQKInPlace(buf: []f32, n_heads: u32, n_cols: usize, scratch: []f32) void {
    const n_rows = buf.len / n_cols;
    const head_dim = n_rows / @as(usize, n_heads);
    const half = head_dim / 2;
    for (0..@as(usize, n_heads)) |h| {
        const head_start = h * head_dim * n_cols;
        const head_rows = buf[head_start..][0 .. head_dim * n_cols];
        @memcpy(scratch[0..head_dim * n_cols], head_rows);
        for (0..half) |i| {
            @memcpy(buf[head_start + (2 * i) * n_cols ..][0..n_cols], scratch[i * n_cols ..][0..n_cols]);
            @memcpy(buf[head_start + (2 * i + 1) * n_cols ..][0..n_cols], scratch[(half + i) * n_cols ..][0..n_cols]);
        }
    }
}

pub fn prepare_tensorsV3(
    allocator: std.mem.Allocator,
    metadata: *Metadata,
    arch: ModelArch,
    safetensors_buffer: []const u8,
    writer: *GGUFWriter,
    out_file: *std.fs.File,
    qtype: QuantType,
) !void {
    const alignment: u64 = 32;

    const block_count_val = metadata.get(allocator, "block_count") orelse {
        return error.MissingBlockCount;
    };
    const block_count: u32 = block_count_val.u32;

    const n_head: u32 = if (metadata.get(allocator, "attention.head_count")) |v| v.u32 else 1;
    const n_kv_head: u32 = if (metadata.get(allocator, "attention.head_count_kv")) |v| v.u32 else n_head;

    var tensor_map = try getTensorNameMap(allocator, arch, block_count);
    const offkeys = try metadata.offsetKeys(allocator);

    const OffsetPatch = struct {
        tensor_index: usize,
        pos_in_file: usize,
        tensor_info: *const TensorInfo,
        name: []const u8,
        actual_dtype: []const u8,
    };
    var offset_patch_list: std.ArrayList(OffsetPatch) = .empty;
    defer offset_patch_list.deinit(allocator);

    // 1) Write tensor headers
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

        try writer.writeU32(@as(u32, @intCast(shape_to_write.len)));
        for (shape_to_write) |dim| {
            try writer.writeU64(dim);
        }

        const is_norm_header = std.mem.endsWith(u8, new_name.?, "_norm.weight");
        // For Q4_K, the innermost dimension (shape_to_write[0]) must be a multiple
        // of 256. When it isn't, fall back to Q8_0 for that tensor so llama.cpp
        // doesn't reject it at load time.
        const row_dim = if (shape_to_write.len > 0) shape_to_write[0] else 1;
        const actual_dtype: []const u8 = if (is_norm_header) "F32" else switch (qtype) {
            .f16  => "F16",
            .q8_0 => "Q8_0",
            .q4_k => if (row_dim % 256 == 0) "Q4_K" else "Q8_0",
        };

        const kind = @as(u32, @intCast(@intFromEnum(try mapDtypeToGGML(actual_dtype))));
        try writer.writeU32(kind);

        const offset_placeholder_pos = writer.position;
        try writer.writeU64(0);

        try offset_patch_list.append(allocator, .{
            .tensor_index = index,
            .pos_in_file = offset_placeholder_pos,
            .tensor_info = tensor,
            .name = new_name.?,
            .actual_dtype = actual_dtype,
        });
    }

    // 2) Align and record data base
    try writer.writePadding(alignment);
    const data_base = writer.position;

    // 3) Write tensor data and patch offsets
    for (offset_patch_list.items) |patch| {
        const tensor = patch.tensor_info;
        const actual_offset = @as(u64, writer.position - data_base);

        // Patch header offset
        const current_pos = writer.position;
        try out_file.seekTo(@intCast(patch.pos_in_file));
        try out_file.writeAll(&std.mem.toBytes(actual_offset));
        try out_file.seekTo(@intCast(current_pos));

        const start = @as(usize, tensor.data_offsets.start);
        const end = @as(usize, tensor.data_offsets.end);
        const tensor_data = safetensors_buffer[start..end];
        const is_norm = std.mem.endsWith(u8, patch.name, "_norm.weight");
        const tensor_type = try ParseTensorType(tensor.dtype);
        const type_size = TypeSize(tensor_type);
        const count = tensor_data.len / type_size;

        if (is_norm) {
            // Gemma stores RMSNorm weights as (γ - 1); all other architectures
            // (LLaMA, Qwen2, …) store γ directly.  Only add the offset for Gemma.
            const norm_offset: f32 = switch (arch) {
                .GEMMA, .GEMMA2, .GEMMA3 => 1.0,
                else => 0.0,
            };

            var buf = try allocator.alloc(f32, count);
            defer allocator.free(buf);

            for (0..count) |i| {
                buf[i] = switch (tensor_type) {
                    .TensorTypeF32  => @as(f32, @bitCast(readU32LE(tensor_data[i * 4 ..][0..4]))) + norm_offset,
                    .TensorTypeF16  => quant.halfToF32(readU16LE(tensor_data[i * 2 ..][0..2])) + norm_offset,
                    .TensorTypeBF16 => bf16ToF32(readU16LE(tensor_data[i * 2 ..][0..2])) + norm_offset,
                    else => return error.UnsupportedDtype,
                };
            }

            try writer.writer.writeAll(@as([*]const u8, @ptrCast(buf.ptr))[0 .. count * @sizeOf(f32)]);
            writer.advance(count * @sizeOf(f32));
        } else {
            const is_q = std.mem.endsWith(u8, patch.name, ".attn_q.weight");
            const is_k = std.mem.endsWith(u8, patch.name, ".attn_k.weight");
            // Only LLaMA-family models use the NeoX-style blocked RoPE that
            // requires this head-interleave permutation.  Gemma uses a different
            // RoPE convention and must NOT be permuted.
            const needs_permute = (is_q or is_k) and switch (arch) {
                .LLAMA, .QWEN2, .QWEN2VL, .QWEN2MOE, .QWEN => true,
                else => false,
            };
            const perm_heads: u32 = if (is_q) n_head else n_kv_head;

            // Use the dtype that was actually written to the header (which may
            // differ from qtype when a Q4_K fallback to Q8_0 was triggered).
            const effective_dtype = try mapDtypeToGGML(patch.actual_dtype);

            if (needs_permute) {
                // Decode to f32, permute rows for NeoX RoPE head interleaving, re-encode.
                // After the header loop, the shape dims are already swapped in-place
                // (shape[0]=original_cols=n_embd, shape[1]=original_rows=heads*head_dim).
                const n_cols: usize = if (tensor.shape.len >= 2) @as(usize, tensor.shape[0]) else 1;
                const n_rows = count / n_cols;
                const head_dim = n_rows / @as(usize, perm_heads);
                var f32_buf = try allocator.alloc(f32, count);
                defer allocator.free(f32_buf);
                for (0..count) |i| {
                    f32_buf[i] = switch (tensor_type) {
                        .TensorTypeF32  => @as(f32, @bitCast(readU32LE(tensor_data[i * 4 ..][0..4]))),
                        .TensorTypeF16  => quant.halfToF32(readU16LE(tensor_data[i * 2 ..][0..2])),
                        .TensorTypeBF16 => bf16ToF32(readU16LE(tensor_data[i * 2 ..][0..2])),
                        else => return error.UnsupportedDtype,
                    };
                }
                // Permute in-place using a scratch buffer sized for one head's rows.
                const scratch = try allocator.alloc(f32, head_dim * n_cols);
                defer allocator.free(scratch);
                permuteQKInPlace(f32_buf, perm_heads, n_cols, scratch);

                switch (effective_dtype) {
                    .TensorTypeF16 => {
                        var f16_buf = try allocator.alloc(u16, count);
                        defer allocator.free(f16_buf);
                        for (0..count) |i| {
                            f16_buf[i] = @bitCast(@as(f16, @floatCast(f32_buf[i])));
                        }
                        try writer.writer.writeAll(@as([*]const u8, @ptrCast(f16_buf.ptr))[0 .. count * 2]);
                        writer.advance(count * 2);
                    },
                    .TensorTypeQ8_0 => {
                        std.debug.assert(count % 32 == 0);
                        const q_buf = try quant.quantizeTensorQ8_0(allocator, f32_buf);
                        defer allocator.free(q_buf);
                        try writer.writer.writeAll(q_buf);
                        writer.advance(q_buf.len);
                    },
                    .TensorTypeQ4_K => {
                        std.debug.assert(count % 256 == 0);
                        const q_buf = try quant.quantizeTensorQ4K(allocator, f32_buf);
                        defer allocator.free(q_buf);
                        try writer.writer.writeAll(q_buf);
                        writer.advance(q_buf.len);
                    },
                    else => return error.UnsupportedDtype,
                }
            } else {
                switch (effective_dtype) {
                    .TensorTypeF16 => switch (tensor_type) {
                        .TensorTypeBF16 => {
                            var buf = try allocator.alloc(u16, count);
                            defer allocator.free(buf);
                            for (0..count) |i| {
                                buf[i] = bf16ToF16(readU16LE(tensor_data[i * 2 ..][0..2]));
                            }
                            try writer.writer.writeAll(@as([*]const u8, @ptrCast(buf.ptr))[0 .. count * 2]);
                            writer.advance(count * 2);
                        },
                        .TensorTypeF16, .TensorTypeF32 => {
                            try writer.writer.writeAll(tensor_data);
                            writer.advance(tensor_data.len);
                        },
                        else => return error.UnsupportedDtype,
                    },
                    .TensorTypeQ8_0, .TensorTypeQ4_K => {
                        var f32_buf = try allocator.alloc(f32, count);
                        defer allocator.free(f32_buf);

                        for (0..count) |i| {
                            f32_buf[i] = switch (tensor_type) {
                                .TensorTypeF32  => @as(f32, @bitCast(readU32LE(tensor_data[i * 4 ..][0..4]))),
                                .TensorTypeF16  => quant.halfToF32(readU16LE(tensor_data[i * 2 ..][0..2])),
                                .TensorTypeBF16 => bf16ToF32(readU16LE(tensor_data[i * 2 ..][0..2])),
                                else => return error.UnsupportedDtype,
                            };
                        }

                        const q_buf = switch (effective_dtype) {
                            .TensorTypeQ8_0 => blk: {
                                std.debug.assert(count % 32 == 0);
                                break :blk try quant.quantizeTensorQ8_0(allocator, f32_buf);
                            },
                            .TensorTypeQ4_K => blk: {
                                std.debug.assert(count % 256 == 0);
                                break :blk try quant.quantizeTensorQ4K(allocator, f32_buf);
                            },
                            else => unreachable,
                        };
                        defer allocator.free(q_buf);

                        try writer.writer.writeAll(q_buf);
                        writer.advance(q_buf.len);
                    },
                    else => return error.UnsupportedDtype,
                }
            }
        }

        // pad to alignment
        try writer.writePadding(alignment);
    }
}
fn bf16ToF16(bf16_bits: u16) u16 {
    // Simple linear conversion: take top 7 exponent + 8 mantissa bits, or implement proper rounding
    const f32_val = bf16ToF32(bf16_bits); // convert BF16 -> F32
    return quant.f32ToHalf(f32_val); // convert F32 -> F16
}
// --- Helper functions for little-endian reads ---
fn readU16LE(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
}

fn readU32LE(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) | (@as(u32, bytes[1]) << 8) | (@as(u32, bytes[2]) << 16) | (@as(u32, bytes[3]) << 24);
}

fn bf16ToF32(bits: u16) f32 {
    const shifted = @as(u32, bits) << 16;
    return @as(f32, @bitCast(shifted));
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
    var offset_patch_list: std.ArrayList(OffsetPatch) = .empty;
    defer offset_patch_list.deinit(allocator);

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

        try offset_patch_list.append(allocator, .{
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
        //603979776
        const start = @as(usize, @intCast(tensor.data_offsets.start));
        const end = @as(usize, @intCast(tensor.data_offsets.end));
        const tensor_data = safetensors_buffer[start..end];
        std.debug.print("Raw data start of {s}: {any}\n", .{ patch.name, safetensors_buffer[start..@min(start + 32, end)] });
        const is_norm = std.mem.endsWith(u8, patch.name, "_norm.weight");
        const tensor_type = try ParseTensorType(tensor.dtype);
        const type_size = TypeSize(tensor_type);
        const count = tensor_data.len / type_size;
        //if (is_norm) {
        // We need to write F32 values (header says F32 for norms). Convert from original dtype -> F32,
        // add 1.0 to each element, then write F32 bytes.
        //const src_dtype = tensor.dtype; // original safetensors dtype string, e.g. "F16", "F32", "BF16"
        var buf = try allocator.alloc(f32, count);
        defer allocator.free(buf);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            switch (tensor_type) {
                .TensorTypeF32 => {
                    // src is f32: read 4 bytes per element, add 1.0
                    const b0 = @as(u32, tensor_data[i * 4 + 0]);
                    const b1 = @as(u32, tensor_data[i * 4 + 1]);
                    const b2 = @as(u32, tensor_data[i * 4 + 2]);
                    const b3 = @as(u32, tensor_data[i * 4 + 3]);
                    const bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    if (is_norm) {
                        buf[i] = @as(f32, @bitCast(bits)) + 1.0;
                    }
                    buf[i] = @as(f32, @bitCast(bits));
                    //std.debug.print("buf F32:{any}", .{buf[i]});

                },
                .TensorTypeF16 => {
                    // src is f16: each element is 2 bytes. Convert half -> f32, add 1.0

                    const lo = @as(u16, tensor_data[i * 2 + 0]);
                    const hi = @as(u16, tensor_data[i * 2 + 1]);
                    const halfBits = lo | (@as(u16, hi) << 8);
                    if (is_norm) {
                        buf[i] = quant.halfToF32(halfBits) + 1.0;
                    }
                    buf[i] = quant.halfToF32(halfBits);
                    //std.debug.print("buf F16:{any}", .{buf[i]});
                },
                .TensorTypeBF16 => {

                    // bfloat16: 2 bytes, top 16 bits of f32
                    // Convert to f32 by shifting left 16 bits and adding 1.0

                    const lo = @as(u32, tensor_data[i * 2 + 0]); // first byte
                    const hi = @as(u32, tensor_data[i * 2 + 1]); // second byte
                    const halfBits = lo | (hi << 8); // little-endian word
                    const bits = halfBits << 16; // shift to top 16 bits of F32
                    if (is_norm) {
                        buf[i] = @as(f32, @bitCast(bits)) + 1.0;
                    }
                    buf[i] = @as(f32, @bitCast(bits));
                    //std.debug.print("buf B16:{any}", .{buf[i]});
                },
                else => return error.UnsupportedDtype,
            }
        }

        try writer.writer.writeAll(@as([*]const u8, @ptrCast(buf.ptr))[0 .. count * @sizeOf(f32)]);
        writer.advance(count * @sizeOf(f32));
        //try writer.writer.writeAll(tensor_data);
        //writer.advance(tensor_data.len);
        //}

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
    arch: ModelArch,
    safetensors_buffer: []const u8,
    basename: []const u8,
    architecture: []const u8,
    model_name: []const u8,
    tok: TokenizerSpec,
    qtype: QuantType,
    out_file: *std.fs.File,
) !void {
    var writer = GGUFWriter.init(out_file.*);

    try writeGGUFHeader(&writer, metadata, tok.kvCount());

    var total_params: u64 = 0;
    for (metadata.tensors.items) |tensor| {
        var n: u64 = 1;
        for (tensor.shape) |d| n *= d;
        total_params += n;
    }

    try writeGeneralMetadata(
        &writer,
        basename,
        architecture,
        model_name,
        total_params,
    );

    try writeExtraMetadataKV(&writer, metadata);

    switch (tok) {
        .sentencepiece => |path| try writeSentencePieceTokenizerVocab(allocator, &writer, path),
        .bpe => |paths| try writeBPETokenizerVocab(allocator, &writer, paths.json, paths.config),
    }

    try writer.writeString("general.quantization_version");
    try writer.writeU32(4); // ggufTypeUint32
    try writer.writeU32(2);

    try prepare_tensorsV3(
        allocator,
        metadata,
        arch,
        safetensors_buffer,
        &writer,
        out_file,
        qtype,
    );
}

fn prepare_metadata(allocator: std.mem.Allocator, metadata: *Metadata, model: registry.ModelInfo, model_files: []const []const u8) !ModelArch {
    var detected_arch: ModelArch = .LLAMA;
    if (utils.indexOfStringInList(model_files, "config.json")) |i| {
        if (try utils.tryLoadJson(allocator, model, model_files[i])) |cfg| {
            const obj = cfg.object;

            // Detect architecture from model_type before any metadata.put() calls
            // so that all KV keys get the correct prefix.
            if (obj.get("model_type")) |v| {
                if (v == .string) {
                    detected_arch = archFromModelType(v.string);
                    metadata.arch_prefix = ModelArchNames.get(detected_arch);
                }
            }

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
                .{ .u32 = 7 }, // LLAMA_FTYPE_MOSTLY_Q8_0
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
    return detected_arch;
}

pub fn convert(model_name: []const u8, output_path: []const u8, qtype: QuantType, allocator: std.mem.Allocator) !void {
    const fs = std.fs;

    const model = try registry.findModelErrorless(model_name);
    const found_model = model.?;
    const model_files = found_model.files;

    std.debug.print("Loading model: {s}\n", .{found_model.name});

    const buffer = try found_model.loadSafetensorsBuffer(allocator);
    defer allocator.free(buffer);

    var metadata = try parseSafetensorsFromBuffer(allocator, model_name, buffer);
    const model_arch = try prepare_metadata(allocator, &metadata, found_model, model_files);
    const arch_str = ModelArchNames.get(model_arch);

    // Override file_type based on selected quant
    try metadata.metadata.put("general.file_type", .{ .u32 = switch (qtype) {
        .f16  => 1,
        .q8_0 => 7,
        .q4_k => 12, // LLAMA_FTYPE_MOSTLY_Q4_K_S
    }});

    var output_file = try fs.cwd().createFile(
        output_path,
        .{ .read = false, .truncate = true },
    );
    defer output_file.close();

    const tok: TokenizerSpec = blk: {
        const sp_path = try found_model.localFilePath(found_model.name, "tokenizer_export.json");
        if (std.fs.cwd().access(sp_path, .{})) |_| {
            break :blk TokenizerSpec{ .sentencepiece = sp_path };
        } else |_| {}

        const bpe_json = try found_model.localFilePath(found_model.name, "tokenizer.json");
        std.fs.cwd().access(bpe_json, .{}) catch return error.NoTokenizerFound;
        const bpe_cfg = try found_model.localFilePath(found_model.name, "tokenizer_config.json");
        break :blk TokenizerSpec{ .bpe = .{ .json = bpe_json, .config = bpe_cfg } };
    };

    try convertToGGUFFromSafeTensors(
        allocator,
        &metadata,
        model_arch,
        buffer,
        found_model.name,
        arch_str,
        found_model.name,
        tok,
        qtype,
        &output_file,
    );

    std.debug.print("✓ Converted '{s}' → '{s}' ({s})\n", .{
        model_name,
        output_path,
        @tagName(qtype),
    });
}
