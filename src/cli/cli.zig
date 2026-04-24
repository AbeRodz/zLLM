const std = @import("std");
const models = @import("../registry/model_registry.zig");
const client = @import("../client/client.zig");
const gguf = @import("../llama/gguf_converter.zig");
const llama = @import("../llama/llama.zig");
const look = @import("../llama/lookahead.zig").look;
const tk = @import("tokamak");
const api = @import("../api/api.zig");
const ggufType = @import("../ggml/gguf.zig");
const safetensors = @import("../safetensors/safetensors.zig");
const converter = @import("../safetensors/gguf/convert.zig");
const session_manager = @import("../inference/session_manager.zig");
const errh = @import("../api/error_handler.zig");

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

/// Pointer to the live server so the signal handler can stop it.
/// Written once before server.start(), never after.
var g_server: ?*tk.Server = null;

fn onSignal(sig: c_int) callconv(.c) void {
    _ = sig;
    // Mark the server as shutting down — acquireSlot() will refuse new work.
    errh.initiateShutdown();
    // stop() signals httpz to exit its accept loop, which causes server.start()
    // to return in the server thread.  deinit() is called later after join().
    if (g_server) |s| s.stop();
}

fn setupSignalHandlers() void {
    const action = std.posix.Sigaction{
        .handler = .{ .handler = onSignal },
        .mask = std.mem.zeroes(std.posix.sigset_t),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.TERM, &action, null);
    std.posix.sigaction(std.posix.SIG.INT, &action, null);
}

fn runServerThread(server: *tk.Server) void {
    server.start() catch |err| {
        // Connection errors during shutdown are expected — only log real ones.
        if (!errh.isShuttingDown()) {
            std.log.err("server fatal error: {}", .{err});
        }
    };
}
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

fn read(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    try ggufType.read(model_name, allocator);
}
fn readSafeTensors(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    try safetensors.read(model_name, allocator);
}
fn convertSafeTensors(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    try converter.convert(model_name, "./model.gguf", allocator);
}

fn ggufInfo(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;

    try ggufType.describe(model_name, allocator);
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
    const prompt = args.next(); // optional — null means interactive stdin loop
    const n_ctx = 8192;

    if (prompt) |p| {
        llama.execute_prompt(model_name, p, n_ctx, allocator) catch |err| {
            std.debug.print("Error during execution: {}\n", .{err});
            return err;
        };
    } else {
        llama.execute(model_name, n_ctx, allocator) catch |err| {
            std.debug.print("Error during execution: {}\n", .{err});
            return err;
        };
    }
}

fn runlookahead(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const model_name = args.next() orelse return error.InvalidUsage;
    const prompt = args.next() orelse "Once upon a time";

    const modelInfo = (try models.findModelErrorless(model_name)) orelse {
        std.debug.print("Unknown model: {s}\n", .{model_name});
        return error.UnknownModel;
    };

    // Prefer an already-converted .gguf; fall back to the default path.
    var gguf_path: []const u8 = try modelInfo.localFilePath(modelInfo.name, "model.gguf");
    for (modelInfo.files) |file| {
        if (std.mem.endsWith(u8, file, ".gguf")) {
            gguf_path = try modelInfo.localFilePath(modelInfo.name, file);
            break;
        }
    }

    look(gguf_path, prompt, allocator) catch |err| {
        std.debug.print("Error during lookahead execution: {}\n", .{err});
        return err;
    };
}
fn serve(args: *std.process.ArgIterator, allocator: std.mem.Allocator) !void {
    const port_str = args.next() orelse "8080";
    const parsedPort = try std.fmt.parseInt(u16, port_str, 10);

    // Session manager must be ready before the first request arrives.
    session_manager.globalInit(
        std.heap.page_allocator,
        32,   // max_sessions  — each holds one llama_context in VRAM
        900,  // ttl_seconds   — idle sessions evicted after 15 minutes
        8192, // n_ctx per session
    );

    // Register SIGTERM / SIGINT handlers before starting the listener so
    // no signal is lost during startup.
    setupSignalHandlers();

    APIPresentation(parsedPort);

    var server = try tk.Server.init(allocator, api.routes, .{ .listen = .{ .port = parsedPort } });
    defer server.deinit();
    g_server = &server;

    // Run the HTTP listener in a background thread so the main thread can
    // handle shutdown coordination without blocking.
    const server_thread = try std.Thread.spawn(.{}, runServerThread, .{&server});

    std.log.info("server listening on port {d} — send SIGTERM or SIGINT to shut down", .{parsedPort});

    // Park the main thread until a signal fires.
    while (!errh.isShuttingDown()) {
        std.Thread.sleep(100 * std.time.ns_per_ms);
    }

    // stop() was already called by the signal handler, which causes
    // server.start() to return so the server thread can exit cleanly.
    server_thread.join();
    g_server = null;

    std.log.info("shutdown: listener stopped — waiting up to 30 s for {d} active request(s)…", .{
        errh.activeRequests(),
    });

    const clean = errh.drainRequests(30);
    if (clean) {
        std.log.info("shutdown: clean exit", .{});
    } else {
        std.log.warn("shutdown: timed out — {d} request(s) still active, forcing exit", .{
            errh.activeRequests(),
        });
    }
}

pub fn init() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
    } else if (std.mem.eql(u8, command, "run-lookahead")) {
        try runlookahead(&args, allocator);
    } else if (std.mem.eql(u8, command, "serve")) {
        try serve(&args, allocator);
    } else if (std.mem.eql(u8, command, "read")) {
        try read(&args, allocator);
    } else if (std.mem.eql(u8, command, "read-safetensors")) {
        try readSafeTensors(&args, allocator);
    } else if (std.mem.eql(u8, command, "convert-safetensors")) {
        try convertSafeTensors(&args, allocator);
    } else if (std.mem.eql(u8, command, "describe")) {
        try ggufInfo(&args, allocator);
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
        \\  get                 Downloads a model from HuggingFace
        \\  convert             Converts a downloaded model to GGUF
        \\  read                Reads, displays and validates a GGUF model info
        \\  read-safetensors    Reads, displays and validates a Safetensors model info
        \\  serve               Serves http server 
        \\  help                Show this message
        \\
        \\Examples:
        \\  zig build run -- get gemma3
        \\  zig build run -- convert gemma3
        \\  zig build run -- serve
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
