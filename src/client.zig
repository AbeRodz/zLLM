const std = @import("std");
const http = std.http;
const Thread = std.Thread;
const ModelInfo = @import("model_registry.zig").ModelInfo;
const ModelRegistry = @import("model_registry.zig");
const progressBar = @import("progress_bar.zig");
const DownloadContext = @import("download_context.zig").DownloadContext;

pub const CHUNK_SIZE = 8 * 1024 * 1024;
pub var NUM_THREADS: usize = 8;

fn buildHeaders(ctx: *DownloadContext, extra: []const http.Header) ![]http.Header {
    var headers = std.ArrayList(http.Header).init(ctx.allocator);
    for (extra) |h| try headers.append(h);
    if (ctx.auth_token) |token| {
        const bearer = try std.fmt.allocPrint(ctx.allocator, "Bearer {s}", .{token});
        try headers.append(.{ .name = "Authorization", .value = bearer });
    }
    return headers.toOwnedSlice();
}

fn do(ctx: *DownloadContext, uri: std.Uri, method: http.Method, headers: []http.Header, buf: []u8) !http.Client.Request {
    var req = try ctx.client.open(method, uri, .{
        .server_header_buffer = buf,
        .extra_headers = headers,
    });
    try req.send();
    try req.wait();
    return req;
}

const RedirectResult = struct {
    url: []const u8,
    required_auth: bool,
};

fn resolveRedirects(ctx: *DownloadContext, start_url: []const u8) !RedirectResult {
    var url_buffer: [4096]u8 = undefined;
    var url = start_url;
    var required_auth = false;

    const uri = try std.Uri.parse(start_url);

    var buf: [4096]u8 = undefined;
    const headers = try buildHeaders(ctx, &[_]http.Header{});
    defer ctx.allocator.free(headers);

    var req = try do(ctx, uri, .HEAD, headers, &buf);
    defer req.deinit();

    const res = req.response;

    if (res.status == .unauthorized) {
        if (ctx.auth_token == null) {
            std.debug.print("repository requires bearer token, please add HF_TOKEN env variable and try again.", .{});
            return error.RequiredBearerToken;
        }
        const bearer = try std.fmt.allocPrint(ctx.allocator, "Bearer {s}", .{ctx.auth_token.?});
        defer ctx.allocator.free(bearer);

        // Retry the request with auth
        var req_retry = try do(ctx, uri, .HEAD, headers, &buf);

        defer req_retry.deinit();

        const res_retry = req_retry.response;

        if (res_retry.status == .found or res_retry.status == .moved_permanently) {
            if (res_retry.location) |new_url| {
                url = std.fmt.bufPrint(&url_buffer, "{s}", .{new_url}) catch return error.UrlTooLong;
                required_auth = true;
            } else {
                return error.MissingRedirectLocation;
            }
        } else {
            std.debug.print("Failed after retry with HTTP status: {}\n", .{res_retry.status});
            return error.HttpError;
        }
    } else if (res.status == .found or res.status == .moved_permanently) {
        if (res.location) |new_url| {
            url = std.fmt.bufPrint(&url_buffer, "{s}", .{new_url}) catch return error.UrlTooLong;
        } else {
            return error.MissingRedirectLocation;
        }
    } else {
        std.debug.print("Failed with HTTP status: {}\n", .{res.status});
        return error.HttpError;
    }
    return RedirectResult{ .url = try std.fmt.allocPrint(ctx.allocator, "{s}", .{url}), .required_auth = required_auth };
}

pub fn downloadChunk(ctx: *DownloadContext, url: []const u8, start: usize, end: usize, file_path: []const u8) !void {
    const uri = try std.Uri.parse(url);
    const range_value = try std.fmt.allocPrint(ctx.allocator, "bytes={d}-{d}", .{ start, end });
    defer ctx.allocator.free(range_value);
    const headers = try buildHeaders(ctx, &[_]http.Header{
        .{ .name = "Range", .value = range_value },
    });
    defer ctx.allocator.free(headers);

    var buf: [4096]u8 = undefined;
    var req = try do(ctx, uri, .GET, headers, &buf);
    defer req.deinit();

    if (req.response.status != .partial_content and req.response.status != .ok) {
        std.debug.print("Failed to fetch chunk {d}-{d}: {d}\n", .{ start, end, @intFromEnum(req.response.status) });
        return error.DownloadFailed;
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
}

pub fn downloadConfigFile(ctx: *DownloadContext, url: []const u8, file_path: []const u8) !void {
    const uri = try std.Uri.parse(url);

    const buf: []u8 = try ctx.allocator.alloc(u8, 4096);
    defer ctx.allocator.free(buf);
    const headers = try buildHeaders(ctx, &[_]http.Header{});
    defer ctx.allocator.free(headers);

    var req = try do(ctx, uri, .GET, headers, buf);
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

pub fn ensureParentDirExists(file_path: []const u8) !void {
    const dir_path = std.fs.path.dirname(file_path).?;

    std.fs.makeDirAbsolute(dir_path) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
}

pub fn parallelDownload(ctx: *DownloadContext, url: []const u8, file_path: []const u8) !void {
    const resolved_redirect = try resolveRedirects(ctx, url);
    const uri = try std.Uri.parse(resolved_redirect.url);
    // Get file size
    const buf: []u8 = try ctx.allocator.alloc(u8, 8192);
    const headers = try buildHeaders(ctx, &[_]http.Header{});
    defer ctx.allocator.free(buf);

    var req = try do(ctx, uri, .HEAD, headers, buf);
    defer req.deinit();
    try req.send();
    try req.wait();

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

    for (thread_list.items) |t| {
        t.join();
    }
    progress_thread.join();

    std.debug.print("Download completed: {s}\n", .{file_path});
}

pub fn downloader(model_info: ModelInfo, allocator: std.mem.Allocator) !void {
    const token = std.process.getEnvVarOwned(allocator, "HF_TOKEN") catch null;
    var ctx = DownloadContext{
        .allocator = allocator,
        .auth_token = token,
        .client = std.http.Client{ .allocator = allocator },
    };
    var backoff_ns: u64 = 0;
    std.debug.print("Downloading model: {s}\n", .{model_info.name});
    for (model_info.files) |file_name| {
        var full_url_buffer: [128]u8 = undefined;
        const full_url = try std.fmt.bufPrint(&full_url_buffer, "{s}{s}?download=true", .{ model_info.base_uri, file_name });

        const file_path = try model_info.localFilePath(model_info.name, file_name);
        try ensureParentDirExists(file_path);
        std.debug.print("Downloading file: {s}\n", .{file_name});

        if (std.mem.endsWith(u8, file_name, ".safetensors") or std.mem.endsWith(u8, file_name, ".gguf")) {
            try parallelDownload(&ctx, full_url, file_path);
            continue;
        }

        downloadConfigFile(&ctx, full_url, file_path) catch |e|
            switch (e) {
                error.PathAlreadyExists => {
                    continue;
                },
                else => return e,
            };
        if (backoff_ns > 0) {
            std.debug.print("Sleeping for {} ns before next config...\n", .{backoff_ns});
            std.time.sleep(backoff_ns);
        }

        backoff_ns += 1_000_000_000; // Add 1s more for each config file as backoff
    }
    std.debug.print("Download completed for {s}\n", .{model_info.name});
}
