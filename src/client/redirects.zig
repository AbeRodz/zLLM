const std = @import("std");
const http = std.http;
const DownloadContext = @import("download_context.zig").DownloadContext;
const utils = @import("utils.zig");

const RedirectResult = struct {
    url: []const u8,
    required_auth: bool,
};

pub fn resolveRedirects(ctx: *DownloadContext, start_url: []const u8) !RedirectResult {
    var url_buffer: [4096]u8 = undefined;
    var url = start_url;
    var required_auth = false;

    const uri = try std.Uri.parse(start_url);

    //var buf: [4096]u8 = undefined;
    const headers = try utils.buildHeaders(ctx, ctx.allocator, &[_]http.Header{});
    defer ctx.allocator.free(headers);
    var client = http.Client{ .allocator = ctx.allocator };
    defer client.deinit();
    var req = try utils.do(&client, uri, .HEAD, headers);
    defer req.deinit();

    const res = try req.receiveHead(&.{});

    if (res.head.status == .unauthorized) {
        if (ctx.auth_token == null) {
            std.debug.print("repository requires bearer token, please add HF_TOKEN env variable and try again.", .{});
            return error.RequiredBearerToken;
        }
        const bearer = try std.fmt.allocPrint(ctx.allocator, "Bearer {s}", .{ctx.auth_token.?});
        defer ctx.allocator.free(bearer);

        // Retry the request with auth

        var req_retry = try utils.do(&client, uri, .HEAD, headers);

        defer req_retry.deinit();

        const res_retry = try req_retry.receiveHead(&.{});

        if (res_retry.head.status == .found or res_retry.head.status == .moved_permanently) {
            if (res_retry.head.location) |new_url| {
                url = std.fmt.bufPrint(&url_buffer, "{s}", .{new_url}) catch return error.UrlTooLong;
                required_auth = true;
            } else {
                return error.MissingRedirectLocation;
            }
        } else {
            std.debug.print("Failed after retry with HTTP status: {}\n", .{res_retry.head.status});
            return error.HttpError;
        }
    } else if (res.head.status == .found or res.head.status == .moved_permanently) {
        if (res.head.location) |new_url| {
            url = std.fmt.bufPrint(&url_buffer, "{s}", .{new_url}) catch return error.UrlTooLong;
        } else {
            return error.MissingRedirectLocation;
        }
    } else {
        std.debug.print("Failed with HTTP status: {}\n", .{res.head.status});
        return error.HttpError;
    }
    return RedirectResult{ .url = try std.fmt.allocPrint(ctx.allocator, "{s}", .{url}), .required_auth = required_auth };
}
