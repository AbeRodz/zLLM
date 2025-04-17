const std = @import("std");

pub const ConvertError = error{
    MissingCacheDir,
    GGUFAlreadyExists,
    ConversionFailed,
};

pub fn convertToGGUF(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    hf_model_path: []const u8,
    gguf_path: []const u8,
    python_bin: []const u8,
    convert_script_path: []const u8,
) !void {
    const gguf_exists = blk: {
        const access_result = std.fs.cwd().access(gguf_path, .{}) catch |err| {
            if (err == error.FileNotFound) break :blk false;
            return err;
        };
        _ = access_result;
        break :blk true;
    };

    if (gguf_exists) {
        std.debug.print("[gguf] GGUF already exists for model: {s}\n", .{model_name});
        return ConvertError.GGUFAlreadyExists;
    }

    std.debug.print("[gguf] Converting model '{s}' from safetensors to GGUF...\n", .{model_name});

    const argv = &[_][]const u8{
        python_bin,
        convert_script_path,
        hf_model_path,
        "--outfile",
        gguf_path,
    };

    var process = std.process.Child.init(argv, allocator);
    process.stdout_behavior = .Inherit;
    process.stderr_behavior = .Inherit;
    process.stdin_behavior = .Inherit;

    const result = try process.spawnAndWait();
    if (result.Exited != 0) {
        std.debug.print("[gguf] Error: conversion script exited with code {}\n", .{result.Exited});
        return ConvertError.ConversionFailed;
    }

    std.debug.print("[gguf] Model successfully converted to GGUF at: {s}\n", .{gguf_path});
}
