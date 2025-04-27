const std = @import("std");
const cli = @import("./cli/cli.zig");
pub fn main() !void {
    cli.init() catch |err| {
        switch (err) {
            error.PreexistingModelFound => {
                std.log.err("Model already downloaded.", .{});
            },
            else => {
                std.log.err("Unexpected error: {}", .{err});
            },
        }
        return;
    };
}
