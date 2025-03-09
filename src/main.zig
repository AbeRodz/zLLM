const std = @import("std");
const models = @import("models.zig");
const client = @import("client.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.skip(); // Skip program name

    const model_name = args.next() orelse {
        std.debug.print("Usage: downloader <model-name> [output-path] [threads]\n", .{});
        return error.InvalidUsage;
    };

    if (args.next()) |num_threads_str| {
        client.NUM_THREADS = try std.fmt.parseInt(usize, num_threads_str, 10);
    }
    const model_info = try models.findModel(model_name) orelse {
        std.debug.print("Unknown model: {s}\n", .{model_name});
        return error.UnknownModel;
    };

    try client.downloader(model_info, allocator);
}
