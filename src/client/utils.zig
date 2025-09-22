const std = @import("std");
const http = std.http;
const DownloadContext = @import("download_context.zig").DownloadContext;

pub fn ensureParentDirExists(file_path: []const u8) !void {
    const dir_path = std.fs.path.dirname(file_path).?;

    std.fs.makeDirAbsolute(dir_path) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
}

pub fn do(client: *http.Client, uri: std.Uri, method: http.Method, headers: []http.Header, buf: []u8) !http.Client.Request {
    var req = try client.open(method, uri, .{
        .server_header_buffer = buf,
        .extra_headers = headers,
    });
    try req.send();
    try req.wait();
    return req;
}

pub fn buildHeaders(ctx: *DownloadContext, allocator: std.mem.Allocator, extra: []const http.Header) ![]http.Header {
    var headers = std.ArrayList(http.Header).init(allocator);
    for (extra) |h| try headers.append(h);
    if (ctx.auth_token) |token| {
        const bearer = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
        try headers.append(.{ .name = "Authorization", .value = bearer });
    }
    return headers.toOwnedSlice();
}
pub fn buildHeadersV2(ctx: *DownloadContext, extra: []const http.Header, allocator: std.mem.Allocator) ![]http.Header {
    // Assume max 16 headers
    var headers_buffer: [16]http.Header = undefined;
    var headers_len: usize = 0;

    for (extra) |h| {
        if (headers_len >= headers_buffer.len) return error.TooManyHeaders;
        headers_buffer[headers_len] = h;
        headers_len += 1;
    }

    if (ctx.auth_token) |token| {
        if (headers_len >= headers_buffer.len) return error.TooManyHeaders;
        const bearer = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
        headers_buffer[headers_len] = .{ .name = "Authorization", .value = bearer };
        headers_len += 1;
    }

    // Allocate a slice to return
    const headers_slice = try allocator.alloc(http.Header, headers_len);
    @memcpy(headers_slice, headers_buffer[0..headers_len]);
    return headers_slice;
}
