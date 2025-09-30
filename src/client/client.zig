const std = @import("std");
const http = std.http;
const Thread = std.Thread;
const ModelInfo = @import("../registry/model_registry.zig").ModelInfo;
const ModelRegistry = @import("../registry/model_registry.zig");
const progressBar = @import("progress_bar.zig");
const DownloadContext = @import("download_context.zig").DownloadContext;
const resolveRedirects = @import("redirects.zig").resolveRedirects;
const utils = @import("utils.zig");

pub const CHUNK_SIZE = 8 * 1024 * 1024;
pub var NUM_THREADS: usize = 8;
pub var MAX_RETRIES: usize = 5;
pub const TIMEOUT_NS: u64 = 30_000_000_000; // 30s timeout

pub fn downloadChunk(ctx: *DownloadContext, url: []const u8, start: usize, end: usize, file_path: []const u8) !void {
    var attempt: usize = 0;
    const uri = try std.Uri.parse(url);
    const range_value = try std.fmt.allocPrint(ctx.allocator, "bytes={d}-{d}", .{ start, end });
    defer ctx.allocator.free(range_value);
    const headers = try utils.buildHeaders(ctx, ctx.allocator, &[_]http.Header{
        .{ .name = "Range", .value = range_value },
    });
    defer for (headers) |h| ctx.allocator.free(h.value); // Free individual header values
    defer ctx.allocator.free(headers);
    while (attempt < MAX_RETRIES) : (attempt += 1) {
        var buf: [4096]u8 = undefined;
        var client = http.Client{ .allocator = ctx.allocator };
        defer client.deinit();
        var req = try utils.do(&ctx.client, uri, .GET, headers, &buf);
        defer req.deinit();

        if (req.response.status != .partial_content and req.response.status != .ok) {
            std.debug.print("Chunk {d}-{d} failed, attempt {d}, HTTP {d}\n", .{ start, end, attempt, @intFromEnum(req.response.status) });
            continue;
        }

        var file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_write });
        defer file.close();

        var reader = req.reader();
        var buffer: [4096]u8 = undefined;
        var offset = start;
        var total_written: usize = 0;

        while (true) {
            const bytes_read = try reader.read(&buffer);
            if (bytes_read == 0) break;

            try file.pwriteAll(buffer[0..bytes_read], offset);
            offset += bytes_read;
            total_written += bytes_read;
            _ = ctx.download_progress.fetchAdd(bytes_read, .seq_cst);
        }
        try file.sync();
        return;
    }
    return error.DownloadFailed;
}

pub fn downloadChunkV3(
    ctx: *DownloadContext,
    url: []const u8,
    start: usize,
    end: usize,
    file_path: []const u8,
) void {
    var attempt: usize = 0;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const thread_allocator = arena.allocator();
    //var arena = std.heap.ArenaAllocator.init(ctx.allocator);
    //defer arena.deinit();

    const uri = std.Uri.parse(url) catch |err| {
        std.debug.print("Failed to parse url: {s} ({})\n", .{ url, err });
        return;
    };

    const range_value = std.fmt.allocPrint(
        thread_allocator,
        "bytes={d}-{d}",
        .{ start, end },
    ) catch |err| {
        std.debug.print("Failed to alloc range header: {}\n", .{err});
        return;
    };
    //defer thread_allocator.free(range_value);

    const headers = &[_]http.Header{.{ .name = "Range", .value = range_value }};

    while (attempt < MAX_RETRIES) : (attempt += 1) {
        if (attempt > 0) {
            const backoff_ns = 500_000_000; // Exponential backoff
            std.debug.print("Retrying chunk {d}-{d} after {d} ms\n", .{ start, end, backoff_ns / 1_000_000 });
            std.time.sleep(backoff_ns);
        }
        const buf = thread_allocator.alloc(u8, 4096) catch {
            std.debug.print("Buffer alloc failed\n", .{});
            return;
        };
        //defer thread_allocator.free(buf);

        var req = ctx.do(uri, .GET, headers, buf) catch |err| {
            std.debug.print("Request failed (attempt {d}): {}\n", .{ attempt, err });
            continue;
        };
        defer req.deinit();

        if (req.response.status != .partial_content and req.response.status != .ok) {
            std.debug.print(
                "Chunk {d}-{d} failed, attempt {d}, HTTP {d}\n",
                .{ start, end, attempt, @intFromEnum(req.response.status) },
            );
            continue;
        }

        var file = std.fs.cwd().openFile(file_path, .{ .mode = .read_write }) catch |err| {
            std.debug.print("Failed to open file {s}: {}\n", .{ file_path, err });
            return;
        };
        defer file.close();

        var reader = req.reader();
        const buffer = thread_allocator.alloc(u8, 4096) catch {
            std.debug.print("Alloc failed for read buffer\n", .{});
            return;
        };
        //defer thread_allocator.free(buffer);

        var offset = start;

        while (true) {
            const bytes_read = reader.read(buffer) catch |err| {
                std.debug.print("Read failed: {}\n", .{err});
                break;
            };
            if (bytes_read == 0) break;

            file.pwriteAll(buffer[0..bytes_read], offset) catch |err| {
                std.debug.print("Write failed: {}\n", .{err});
                break;
            };
            offset += bytes_read;
            _ = ctx.download_progress.fetchAdd(bytes_read, .seq_cst);
        }

        file.sync() catch |err| {
            std.debug.print("Sync failed: {}\n", .{err});
            // continue to retry
            continue;
        };

        return; // success
    }

    std.debug.print("Chunk {d}-{d} failed after {d} attempts\n", .{ start, end, MAX_RETRIES });
}

pub fn downloadConfigFile(ctx: *DownloadContext, url: []const u8, file_path: []const u8) !void {
    const uri = try std.Uri.parse(url);

    const buf: []u8 = try ctx.allocator.alloc(u8, 4096);
    defer ctx.allocator.free(buf);

    var req = try ctx.do(uri, .GET, &.{}, buf);
    defer req.deinit();
    if (req.response.status == .unauthorized) {
        if (ctx.auth_token == null) {
            std.debug.print("repository requires bearer token, please add HF_TOKEN env variable and try again.", .{});
            return error.RequiredBearerToken;
        }
    }
    if (req.response.status != .ok) {
        std.debug.print("Failed to download config: {s}, HTTP Status: {}\n", .{ file_path, req.response.status });
        return error.DownloadFailed;
    }

    var file = std.fs.cwd().createFile(file_path, .{ .exclusive = true }) catch |e|
        switch (e) {
            error.PathAlreadyExists => {
                std.log.info("already exists", .{});
                return e;
            },
            else => return e,
        };
    defer file.close();

    var reader = req.reader();
    var buffer: [1024]u8 = undefined;

    while (true) {
        const bytes_read = try reader.read(&buffer);
        if (bytes_read == 0) break;
        _ = try file.write(buffer[0..bytes_read]);
    }

    std.debug.print("Config file downloaded: {s}\n", .{file_path});
}

pub fn parallelDownloadV2(ctx: *DownloadContext, url: []const u8, file_path: []const u8) !void {
    const resolved_redirect = try resolveRedirects(ctx, url);
    const uri = try std.Uri.parse(resolved_redirect.url);
    // Get file size
    const buf: []u8 = try ctx.allocator.alloc(u8, 8192);

    var req = try ctx.do(uri, .HEAD, &.{}, buf);
    defer req.deinit();

    const content_length = req.response.content_length orelse return error.MissingContentLength;
    const file_size_str = progressBar.formatBytes(content_length);
    std.debug.print("📦 File size: {s}\n", .{file_size_str});

    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    try file.setEndPos(content_length);

    const start_time = std.time.nanoTimestamp();
    const progress_thread = try Thread.spawn(.{}, progressBar.runProgressBar, .{ ctx, content_length, start_time });
    // Use dynamic thread list
    var thread_list = std.ArrayList(Thread).init(ctx.allocator);
    defer thread_list.deinit();

    const chunk_size = CHUNK_SIZE;

    var offset: usize = 0;
    var pool: Thread.Pool = undefined;
    //const n_jobs = content_length / chunk_size;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const pool_allocator = arena.allocator();

    try pool.init(.{ .allocator = pool_allocator, .n_jobs = NUM_THREADS });
    defer pool.deinit();
    std.debug.print("downloading...", .{});
    var wg: std.Thread.WaitGroup = .{};

    while (offset < content_length) {
        const start = offset;
        const end = @min(start + chunk_size - 1, content_length - 1);
        std.debug.print("dispatching chunk:{d}-{d}\n", .{ start, end });
        pool.spawnWg(&wg, downloadChunkV3, .{ ctx, resolved_redirect.url, start, end, file_path });

        offset += chunk_size;
    }
    wg.wait();

    // for (thread_list.items) |t| {
    //     t.join();
    // }
    progress_thread.join();

    std.debug.print("Download completed: {s}\n", .{file_path});
}

pub fn parallelDownload(ctx: *DownloadContext, url: []const u8, file_path: []const u8) !void {
    const resolved_redirect = try resolveRedirects(ctx, url);
    const uri = try std.Uri.parse(resolved_redirect.url);

    // Get content length
    const buf = try ctx.allocator.alloc(u8, 8192);
    defer ctx.allocator.free(buf);
    const headers = try utils.buildHeaders(ctx, ctx.allocator, &[_]http.Header{});
    var req = try utils.do(&ctx.client, uri, .HEAD, headers, buf);
    defer req.deinit();
    try req.send();
    try req.wait();

    const content_length = req.response.content_length orelse return error.MissingContentLength;
    std.debug.print("📦 File size: {s}\n", .{progressBar.formatBytes(content_length)});

    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    try file.setEndPos(content_length);

    // Spawn progress thread
    const start_time = std.time.nanoTimestamp();
    const progress_thread = try Thread.spawn(.{}, progressBar.runProgressBar, .{ ctx, content_length, start_time });

    var thread_list = std.ArrayList(Thread).init(ctx.allocator);
    defer thread_list.deinit();

    var offset: usize = 0;
    const chunk_size = CHUNK_SIZE;

    while (offset < content_length) {
        const start = offset;
        const end = @min(start + chunk_size - 1, content_length - 1);

        const thread = Thread.spawn(.{}, downloadChunk, .{ ctx, resolved_redirect.url, start, end, file_path }) catch |err| {
            std.debug.print("Thread spawn failed: {}\n", .{err});
            return err;
        };
        try thread_list.append(thread);

        offset += chunk_size;

        if (thread_list.items.len >= NUM_THREADS) {
            for (thread_list.items) |t| t.join();
            thread_list.clearRetainingCapacity();
        }
    }

    for (thread_list.items) |t| t.join();
    progress_thread.join();

    std.debug.print("Download completed: {s}\n", .{file_path});
}

pub fn downloader(model_info: ModelInfo, allocator: std.mem.Allocator) !void {
    const token = std.process.getEnvVarOwned(allocator, "HF_TOKEN") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => {
            std.debug.print("repository requires bearer token, please add HF_TOKEN env variable and try again.\n", .{});
            return error.RequiredBearerToken; // propagate from enclosing fn
        },
        else => return err,
    };
    var ctx = DownloadContext{
        .allocator = std.heap.page_allocator,
        .auth_token = token,
        .client = std.http.Client{ .allocator = allocator },
    };

    std.debug.print("Downloading model: {s}\n", .{model_info.name});
    for (model_info.files) |file_name| {
        var full_url_buffer: [128]u8 = undefined;
        const full_url = try std.fmt.bufPrint(&full_url_buffer, "{s}{s}?download=true", .{ model_info.base_uri, file_name });

        const file_path = try model_info.localFilePath(model_info.name, file_name);
        try utils.ensureParentDirExists(file_path);
        std.debug.print("Downloading file: {s}\n", .{file_name});

        const is_large_file =
            std.mem.endsWith(u8, file_name, ".safetensors") or
            std.mem.endsWith(u8, file_name, ".gguf");

        if (is_large_file) {
            try parallelDownloadV2(&ctx, full_url, file_path);
            continue;
        }
        var attempt: usize = 0;
        var backoff_ns: u64 = 500_000_000;
        while (attempt < MAX_RETRIES) : (attempt += 1) {
            backoff_ns += 1_000_000_000; // Add 1s more for each config file as backoff

            downloadConfigFile(&ctx, full_url, file_path) catch |e|
                switch (e) {
                    error.PathAlreadyExists => break, // skip existing
                    else => {
                        std.debug.print("Config download failed for {s}, attempt {d}\n", .{ file_name, attempt + 1 });
                        std.time.sleep(backoff_ns);
                        backoff_ns *= 2;
                        if (attempt + 1 == MAX_RETRIES) return;
                    },
                };
        }
    }
    std.debug.print("Download completed for {s}\n", .{model_info.name});
}
