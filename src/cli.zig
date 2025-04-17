const std = @import("std");
const models = @import("model_registry.zig");
const client = @import("client.zig");
const gguf = @import("./llama/gguf_converter.zig");

fn get(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    const threads = try getOptionalThreadArg(args);

    if (threads) |n| client.NUM_THREADS = n;

    const model = models.findModel(model_name) catch |err| {
        if (err == error.PreexistingModelFound) {
            std.debug.print("Model already downloaded: {s}\n", .{model_name});
            return err;
        }
        return err;
    };

    if (model == null) {
        std.debug.print("Unknown model: {s}\n", .{model_name});
        return error.UnknownModel;
    }
    try client.downloader(model.?, allocator);
}

fn convert(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;

    const model = models.findModelErrorless(model_name) catch |err| {
        if (err == error.PreexistingModelFound) {
            std.debug.print("Model already downloaded: {s}\n", .{model_name});
            return null;
        }
        return err;
    };

    const cache_dir = try models.getCacheDir(allocator);
    defer allocator.free(cache_dir);

    const model_dir = try std.fs.path.join(allocator, &.{ cache_dir, model.?.name });
    defer allocator.free(model_dir);

    const gguf_path = try std.fs.path.join(allocator, &.{ model_dir, "model.gguf" });
    defer allocator.free(gguf_path);

    try gguf.convertToGGUF(
        allocator,
        model.?.name,
        model_dir,
        gguf_path,
        "venv/bin/python3",

        "llama.cpp/convert_hf_to_gguf.py",
    );
}

pub fn run() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.skip();

    const command = args.next() orelse {
        printUsage();
        return error.InvalidUsage;
    };

    try models.checkDir(allocator);

    if (std.mem.eql(u8, command, "get")) {
        try get(&args, allocator);
    } else if (std.mem.eql(u8, command, "convert")) {
        try convert(&args, allocator);
    } else if (std.mem.eql(u8, command, "help")) {
        printUsage();
    } else if (std.mem.eql(u8, command, "run")) {
        // TODO implement model execution
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        printUsage();
        return error.InvalidUsage;
    }
}

fn getOptionalThreadArg(args: *std.process.ArgIterator) !?usize {
    if (args.next()) |t| {
        const parsed = try std.fmt.parseInt(usize, t, 10);
        return parsed;
    }
    return null;
}

fn printUsage() void {
    std.debug.print(
        \\Usage:
        \\  zig build run -- <command> <model-name> [threads]
        \\
        \\Commands:
        \\  download   Downloads a model from HuggingFace
        \\  convert    Converts a downloaded model to GGUF
        \\  help       Show this message
        \\
        \\Examples:
        \\  zig build run -- download vit-base 8
        \\  zig build run -- convert vit-base
        \\
    , .{});
}
