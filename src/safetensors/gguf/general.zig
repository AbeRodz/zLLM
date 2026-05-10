const std = @import("std");
const GGUFWriter = @import("../../ggml/writer.zig").GGUFWriter;
const parseTokenizerJson = @import("../../ggml/tokenizer.zig").parseTokenizerJsonV2;
const parseBPETokenizerJson = @import("../../ggml/tokenizer.zig").parseBPETokenizerJson;

pub fn modelWeightCountRoundedNotation(
    comptime min_digits: usize,
    model_params_count: u64,
    buf: []u8,
) ![]const u8 {
    var scaled_model_params: f64 = 0;
    var scale_suffix: []const u8 = "";

    if (model_params_count >= 1_000_000_000_000) {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-12;
        scale_suffix = "T";
    } else if (model_params_count >= 1_000_000_000) {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-9;
        scale_suffix = "B";
    } else if (model_params_count >= 1_000_000) {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-6;
        scale_suffix = "M";
    } else {
        scaled_model_params = @as(f64, @floatFromInt(model_params_count)) * 1e-3;
        scale_suffix = "K";
    }

    _ = min_digits;
    var rounded: u64 = @intFromFloat(@round(scaled_model_params));
    // Re-bucket if rounding pushes us to the next tier (e.g. 1000M → 1B).
    if (std.mem.eql(u8, scale_suffix, "M") and rounded >= 1000) {
        rounded = @intFromFloat(@round(scaled_model_params / 1000.0));
        scale_suffix = "B";
    } else if (std.mem.eql(u8, scale_suffix, "B") and rounded >= 1000) {
        rounded = @intFromFloat(@round(scaled_model_params / 1000.0));
        scale_suffix = "T";
    }
    return std.fmt.bufPrint(buf, "{d}{s}", .{ rounded, scale_suffix });
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
    const bos = tokens.special_tokens.get("bos").?;
    {
        try writer.writeString("tokenizer.ggml.bos_token_id");
        try writer.writeU32(4); // ggufTypeUInt32
        try writer.writeU32(@as(u32, @intCast(bos)));
    }
    const eos = tokens.special_tokens.get("eos").?;
    {
        try writer.writeString("tokenizer.ggml.eos_token_id");
        try writer.writeU32(4); // ggufTypeUInt32
        try writer.writeU32(@as(u32, @intCast(eos)));
    }
    const unk = tokens.special_tokens.get("unk").?;
    {
        try writer.writeString("tokenizer.ggml.unknown_token_id");
        try writer.writeU32(4); // ggufTypeUInt32
        try writer.writeU32(@as(u32, @intCast(unk)));
    }
    const pad = tokens.special_tokens.get("pad").?;
    {
        try writer.writeString("tokenizer.ggml.padding_token_id");
        try writer.writeU32(4); // ggufTypeUInt32
        try writer.writeU32(@as(u32, @intCast(pad)));
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

    try writer.writeString("tokenizer.ggml.add_space_prefix");
    try writer.writeU32(7); // ggufTypeArray
    try writer.writeBool(false); // ggufTypeArray

    std.debug.print("Tokenizer vocab set with {d} tokens\n", .{pieces.len});
}

/// Write exactly 9 tokenizer KV entries for a HuggingFace BPE model.
/// Reads both tokenizer.json and tokenizer_config.json from the model directory.
pub fn writeBPETokenizerVocab(
    allocator: std.mem.Allocator,
    writer: *GGUFWriter,
    tokenizer_json_path: []const u8,
    tokenizer_config_path: []const u8,
) !void {
    var data = try parseBPETokenizerJson(allocator, tokenizer_json_path, tokenizer_config_path);
    defer data.deinit();

    // 1. tokenizer.ggml.model = "gpt2"
    try writer.writeString("tokenizer.ggml.model");
    try writer.writeU32(8);
    try writer.writeString("gpt2");

    // 2. tokenizer.ggml.tokens
    try writer.writeString("tokenizer.ggml.tokens");
    try writer.writeU32(9);
    try writer.writeU32(8);
    try writer.writeU64(@as(u64, data.tokens.len));
    for (data.tokens) |tok| try writer.writeString(tok);

    // 3. tokenizer.ggml.token_type
    try writer.writeString("tokenizer.ggml.token_type");
    try writer.writeU32(9);
    try writer.writeU32(5); // int32
    try writer.writeU64(@as(u64, data.token_types.len));
    for (data.token_types) |tt| try writer.writeU32(tt);

    // 4. tokenizer.ggml.merges
    try writer.writeString("tokenizer.ggml.merges");
    try writer.writeU32(9);
    try writer.writeU32(8);
    try writer.writeU64(@as(u64, data.merges.len));
    for (data.merges) |m| try writer.writeString(m);

    // 5. tokenizer.ggml.bos_token_id
    try writer.writeString("tokenizer.ggml.bos_token_id");
    try writer.writeU32(4);
    try writer.writeU32(data.bos_token_id);

    // 6. tokenizer.ggml.eos_token_id
    try writer.writeString("tokenizer.ggml.eos_token_id");
    try writer.writeU32(4);
    try writer.writeU32(data.eos_token_id);

    // 7. tokenizer.ggml.unknown_token_id
    try writer.writeString("tokenizer.ggml.unknown_token_id");
    try writer.writeU32(4);
    try writer.writeU32(data.unk_token_id);

    // 8. tokenizer.ggml.padding_token_id
    try writer.writeString("tokenizer.ggml.padding_token_id");
    try writer.writeU32(4);
    try writer.writeU32(data.pad_token_id);

    // 9. tokenizer.chat_template
    try writer.writeString("tokenizer.chat_template");
    try writer.writeU32(8);
    try writer.writeString(data.chat_template);

    std.debug.print("BPE vocab: {d} tokens, {d} merges, bos={d} eos={d}\n", .{
        data.tokens.len, data.merges.len, data.bos_token_id, data.eos_token_id,
    });
}

pub fn writeGeneralMetadata(
    writer: *GGUFWriter,
    basename: []const u8,
    architecture: []const u8,
    model_name: []const u8,
    total_params: u64,
) !void {
    var output: [64]u8 = undefined;
    const label = try sizeLabel(2, total_params, 0, 0, 0, &output);
    _ = basename;
    const general_tags = .{
        .{ "general.architecture", GeneralTag{ .str = architecture } },
        .{ "general.type", GeneralTag{ .str = "model" } },
        .{ "general.name", GeneralTag{ .str = model_name } },
        .{ "general.size_label", GeneralTag{ .str = label } },
        //.{ "general.file_type", GeneralTag{ .u32 = 1 } },
        //.{ "general.quantization_version", GeneralTag{ .u32 = quant_version } },
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
