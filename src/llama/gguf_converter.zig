const std = @import("std");

pub const ConvertError = error{
    MissingCacheDir,
    GGUFAlreadyExists,
    ConversionFailed,
    MissingVirtualEnv,
    MissingPackages,
};

pub fn convertToGGUF(
    allocator: std.mem.Allocator,
    model_name: []const u8,
    hf_model_path: []const u8,
    gguf_path: []const u8,
    python_bin: []const u8,
    convert_script_path: []const u8,
) !void {
    // Check if GGUF file already exists
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

    // Check if Python is running in a virtual environment
    const check_venv = &[_][]const u8{
        python_bin,
        "-c",
        "import sys; exit(0) if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else exit(1)",
    };

    var venv_check = std.process.Child.init(check_venv, allocator);
    const venv_result = try venv_check.spawnAndWait();
    if (venv_result.Exited != 0) {
        std.debug.print("[gguf] Error: Python binary is not running inside a virtual environment.\n", .{});
        std.debug.print("[gguf] Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r ./llama.cpp/requirements.txt\n", .{});
        return ConvertError.MissingVirtualEnv;
    }

    // Check if required Python packages are installed
    const import_check = &[_][]const u8{
        python_bin,
        "-c",
        "import transformers, safetensors", // Add any other required packages here
    };

    var package_check = std.process.Child.init(import_check, allocator);
    const package_result = try package_check.spawnAndWait();
    if (package_result.Exited != 0) {
        std.debug.print("[gguf] Error: Required Python packages (transformers, safetensors) are missing.\n", .{});
        std.debug.print("[gguf] Please run: pip install -r ./llama.cpp/requirements.txt\n", .{});
        return ConvertError.MissingPackages;
    }

    // Proceed with the conversion
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
