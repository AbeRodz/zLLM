const std = @import("std");
const http = std.http;
const Thread = std.Thread;
const ModelInfo = @import("models.zig").ModelInfo;

pub const CHUNK_SIZE = 4 * 1024 * 1024; // 4MB chunks for parallel download
pub var NUM_THREADS: usize = 4; // Adjust based on network and CPU

fn resolveRedirects(allocator: std.mem.Allocator, start_url: []const u8) ![]const u8 {
    var url_buffer: [4096]u8 = undefined;
    var url = start_url;

    const uri = try std.Uri.parse(start_url);
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();
    var buf: [4096]u8 = undefined;
    var req = try client.open(.HEAD, uri, .{ .server_header_buffer = &buf }); // Use HEAD to avoid downloading
    defer req.deinit();

    try req.send();
    try req.wait();

    const res = req.response;
    if (res.status == .found or res.status == .moved_permanently) { // 302 or 301
        if (res.location) |new_url| {
            std.debug.print("Redirecting to: {s}\n", .{new_url});
            url = std.fmt.bufPrint(&url_buffer, "{s}", .{new_url}) catch return error.UrlTooLong;
        } else {
            return error.MissingRedirectLocation;
        }
    }

    if (res.status != .found) {
        std.debug.print("Failed with HTTP status: {}\n", .{res.status});
        return error.HttpError;
    }

    return std.fmt.allocPrint(allocator, "{s}", .{url});
}

pub fn downloadChunk(url: []const u8, start: usize, end: usize, file_path: []const u8) !void {
    var client = http.Client{ .allocator = std.heap.page_allocator };
    defer client.deinit();
    const uri = try std.Uri.parse(url);
    const range_value = try std.fmt.allocPrint(std.heap.page_allocator, "bytes={d}-{d}", .{ start, end });
    defer std.heap.page_allocator.free(range_value);

    const reqHeaders = [_]std.http.Header{
        .{ .name = "Range", .value = range_value },
    };

    var buf: [4096]u8 = undefined;
    var req = try client.open(.GET, uri, .{
        .server_header_buffer = &buf,
        .extra_headers = &reqHeaders,
    });
    defer req.deinit();

    try req.send();
    try req.wait();

    // Print response status
    std.debug.print("Chunk {d}-{d}: HTTP Status = {d}\n", .{ start, end, @intFromEnum(req.response.status) });

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

        // Print how many bytes are read
        //std.debug.print("Read {d} bytes from {d}-{d}\n", .{ bytes_read, start, end });

        try file.pwriteAll(buffer[0..bytes_read], offset);
        offset += bytes_read;
        total_written += bytes_read;
    }

    // Print final written data
    //std.debug.print("Chunk {d}-{d}: Wrote {d} bytes\n", .{ start, end, total_written });

    try file.sync();
}

pub fn downloadConfigFile(allocator: std.mem.Allocator, url: []const u8, file_path: []const u8) !void {
    var client = http.Client{ .allocator = allocator };
    defer client.deinit();

    //const resolved_url = try resolveRedirects(std.heap.page_allocator, url);
    const uri = try std.Uri.parse(url);

    const server_header_buffer: []u8 = try allocator.alloc(u8, 1024 * 8);
    defer allocator.free(server_header_buffer);

    var req = try client.open(.GET, uri, .{ .server_header_buffer = server_header_buffer });
    defer req.deinit();
    try req.send();
    try req.wait();

    if (req.response.status != .ok) {
        std.debug.print("Failed to download config: {s}, HTTP Status: {}\n", .{ file_path, req.response.status });
        return error.DownloadFailed;
    }

    var file = try std.fs.cwd().createFile(file_path, .{});
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

pub fn parallelDownload(allocator: std.mem.Allocator, url: []const u8, file_path: []const u8) !void {
    var client = http.Client{ .allocator = allocator };
    defer client.deinit();
    const resolved_url = try resolveRedirects(std.heap.page_allocator, url);
    const uri = try std.Uri.parse(resolved_url);
    // Get file size
    const server_header_buffer: []u8 = try allocator.alloc(u8, 1024 * 8);
    defer allocator.free(server_header_buffer);

    var req = try client.open(.HEAD, uri, .{ .server_header_buffer = server_header_buffer });
    defer req.deinit();
    try req.send();
    try req.wait();

    const content_length = req.response.content_length orelse return error.MissingContentLength;
    std.debug.print("File size: {d} bytes\n", .{content_length});

    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    try file.setEndPos(content_length);

    // Use dynamic thread list
    var thread_list = std.ArrayList(Thread).init(allocator);
    defer thread_list.deinit();

    const chunk_size = CHUNK_SIZE;

    for (0..NUM_THREADS) |i| {
        const start = i * chunk_size;
        const end = @min(start + chunk_size - 1, content_length - 1);
        if (start > content_length) break;

        const thread = Thread.spawn(.{}, downloadChunk, .{ url, start, end, file_path }) catch |err| {
            std.debug.print("Thread spawn failed: {}\n", .{err});
            return err;
        };
        try thread_list.append(thread);
    }

    for (thread_list.items) |t| {
        t.join();
    }

    std.debug.print("Download completed: {s}\n", .{file_path});
}

pub fn downloader(model_info: ModelInfo, allocator: std.mem.Allocator) !void {
    for (model_info.files) |file_name| {
        var full_url_buffer: [256]u8 = undefined;
        const full_url = try std.fmt.bufPrint(&full_url_buffer, "{s}{s}", .{ model_info.base_uri, file_name });

        const file_path = std.fs.path.basename(file_name);
        std.debug.print("Downloading: {s}\n", .{file_path});
        if (std.mem.endsWith(u8, file_name, ".safetensors?download=true")) {
            try parallelDownload(allocator, full_url, file_path);
            continue;
        } else {
            try downloadConfigFile(allocator, full_url, file_path);
        }
    }
    std.debug.print("Download completed for {s}\n", .{model_info.name});
}
