const std = @import("std");
const GGUFWriter = @import("../../ggml/writer.zig").GGUFWriter;
const parseTokenizerJson = @import("../../ggml/tokenizer.zig").parseTokenizerJson;

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

pub fn writeGeneralMetadata(
    writer: *GGUFWriter,
    basename: []const u8,
    architecture: []const u8,
    model_name: []const u8,
    //quant_version: u32,
) !void {
    var output: [64]u8 = undefined;
    // TODO Get real parameters count dynamically
    const label = try sizeLabel(2, 1_000_000_000, 0, 0, 0, &output);
    std.debug.print("Label: {s}\n", .{label});
    _ = model_name;
    _ = basename;
    const general_tags = .{
        .{ "general.architecture", GeneralTag{ .str = architecture } },
        .{ "general.type", GeneralTag{ .str = "model" } },
        .{ "general.name", GeneralTag{ .str = "Gemma3" } }, // change
        .{ "general.basename", GeneralTag{ .str = "gemma-3" } },
        .{ "general.size_label", GeneralTag{ .str = "1000M" } }, // should be 1B
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
