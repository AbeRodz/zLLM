const std = @import("std");
const Thread = std.Thread;
const http = std.http;
const DownloadContext = @import("download_context.zig").DownloadContext;

pub fn formatBytes(bytes: usize) []const u8 {
    const mib = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
    return std.fmt.allocPrint(std.heap.page_allocator, "{d:.2} MiB", .{mib}) catch "?.?? MiB";
}

pub fn runProgressBar(ctx: *DownloadContext, total: usize, start_ns: i128) void {
    const bar_width = @min(80, 100) - 40; // must check how to make this dynamic
    const SPINNER_FRAMES = [_][]const u8{ "|", "/", "-", "\\" };
    var tick: usize = 0;

    while (true) {
        const now = std.time.nanoTimestamp();
        const elapsed_ns = now - start_ns;
        const downloaded = ctx.download_progress.load(.seq_cst);

        const percent = downloaded * 100 / total;
        const filled = percent * bar_width / 100;
        const empty = bar_width - filled;

        const speed = if (elapsed_ns > 0)
            downloaded * 1_000_000_000 / @as(usize, @intCast(elapsed_ns))
        else
            0;
        const speed_mbps = speed / (1024 * 1024);
        const eta = if (speed > 0)
            (total - downloaded) / speed
        else
            0;

        const spinner = SPINNER_FRAMES[tick % SPINNER_FRAMES.len];
        tick += 1;

        const downloaded_mib = formatBytes(downloaded);
        const total_mib = formatBytes(total);

        std.debug.print(
            "\r{s} [{s}{s}] {d}% | {s}/{s} | ⏳ {d}s | ⚡ {d} MB/s    ",
            .{
                spinner,
                repeatChar("█", filled),
                repeatChar(" ", empty),
                percent,
                downloaded_mib,
                total_mib,
                eta,
                speed_mbps,
            },
        );

        if (downloaded >= total) break;
        std.time.sleep(100 * std.time.ns_per_ms);
    }

    std.debug.print("\n", .{});
}

fn repeatChar(char_str: []const u8, count: usize) []const u8 {
    var list = std.ArrayList(u8).init(std.heap.page_allocator);
    for (0..count) |_| {
        _ = list.appendSlice(char_str) catch {};
    }
    return list.toOwnedSlice() catch " ";
}
