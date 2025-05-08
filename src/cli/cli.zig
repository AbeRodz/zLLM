const std = @import("std");
const models = @import("../registry/model_registry.zig");
const client = @import("../client/client.zig");
const gguf = @import("../llama/gguf_converter.zig");
const llama = @import("../llama/llama.zig");
const tk = @import("tokamak");
const api = @import("../api/api.zig");

fn get(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    const threads = try getOptionalThreadArg(args);

    if (threads) |n| client.NUM_THREADS = n;

    const model = models.findModel(model_name) catch |err| {
        if (err == error.PreexistingModelFound) {
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

fn run(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    const n_ctx = 8192;
    try llama.execute(model_name, n_ctx, allocator);
}

fn ApiRun(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const port_str = args.next() orelse "8080";
    const parsedPort = try std.fmt.parseInt(u16, port_str, 10);
    APIPresentation(parsedPort);
    var server = try tk.Server.init(allocator, api.routes, .{ .listen = .{ .port = parsedPort } });
    try server.start();
}

pub fn init() !void {
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
        try run(&args, allocator);
    } else if (std.mem.eql(u8, command, "api-run")) {
        try ApiRun(&args, allocator);
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
        \\  get   Downloads a model from HuggingFace
        \\  convert    Converts a downloaded model to GGUF
        \\  help       Show this message
        \\
        \\Examples:
        \\  zig build run -- download vit-base 8
        \\  zig build run -- convert vit-base
        \\
    , .{});
}

pub fn APIPresentation(port: u16) void {
    const YEL = "\x1b[33m";
    const RED = "\x1b[31m";
    const GRN = "\x1b[32m";
    const RESET = "\x1b[0m";

    std.debug.print(YEL ++
        "          __     __     __  ___\n" ++
        " ____   / /    / /    /  |/  /\n" ++
        "/_  /  / /    / /    / /|_/ / \n" ++
        " / /_ / /___ / /___ / /  / /  \n" ++
        "/___//_____//_____//_/  /_/   \n" ++
        "                              \n" ++ RESET, .{});

    std.debug.print(RED ++
        "Fast, portable and lightweight inference server!\n" ++ RESET, .{});

    std.debug.print("Server running on port: " ++ GRN ++ "{d}\n" ++ RESET, .{port});
}
