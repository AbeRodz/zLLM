const std = @import("std");
const models = @import("model_registry.zig");
const client = @import("client.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.skip();
    try models.checkDir(allocator);
    const model_name = args.next() orelse {
        std.debug.print("Usage: downloader <model-name> [output-path] [threads]\n", .{});
        return error.InvalidUsage;
    };

    if (args.next()) |num_threads_str| {
        client.NUM_THREADS = try std.fmt.parseInt(usize, num_threads_str, 10);
    }
    const model_info = models.findModel(model_name) catch |err| {
        if (err == error.PreexistingModelFound) {
            std.debug.print("Model already downloaded: {s}\n", .{model_name});
            return err;
        }
        return err;
    };

    if (model_info == null) {
        std.debug.print("Unknown model: {s}\n", .{model_name});
        return error.UnknownModel;
    }

    try client.downloader(model_info.?, allocator);
}
