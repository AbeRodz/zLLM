const std = @import("std");
const http = std.http;

pub const DownloadContext = struct {
    allocator: std.mem.Allocator,
    client: http.Client,
    auth_token: ?[]const u8 = null,
    download_progress: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn init(allocator: std.mem.Allocator, auth_token: ?[]const u8) DownloadContext {
        return DownloadContext{
            .allocator = allocator,
            .client = http.Client{ .allocator = allocator },
            .auth_token = auth_token,
        };
    }

    pub fn deinit(self: *DownloadContext) void {
        self.client.deinit();
    }

    /// Builds headers with Authorization automatically
    fn buildHeaders(self: *DownloadContext, extra: []const http.Header) ![]http.Header {
        var headers = std.ArrayList(http.Header).init(self.allocator);
        for (extra) |h| try headers.append(h);

        if (self.auth_token) |token| {
            const bearer = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
            try headers.append(.{ .name = "Authorization", .value = bearer });
        }

        return headers.toOwnedSlice();
    }

    /// Perform a request, always injecting Authorization header if needed
    pub fn do(self: *DownloadContext, uri: std.Uri, method: http.Method, extra: []const http.Header, buf: []u8) !http.Client.Request {
        const headers = try self.buildHeaders(extra);
        //defer for (headers) |h| self.allocator.free(h.value); // free header values

        defer self.allocator.free(headers);

        var req = try self.client.open(method, uri, .{
            .server_header_buffer = buf,
            .extra_headers = headers,
        });
        try req.send();
        // req.send() catch |err| {
        //     switch (err) {
        //         error.RequestFailed => {
        //             std.debug.print("Request failed to send\n", .{});
        //         },
        //         else => {
        //             std.debug.print("Request send error: {}\n", .{err});
        //         },
        //     }
        //     return err;
        // };

        try req.wait();
        return req;
    }
};
