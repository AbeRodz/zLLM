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
};
