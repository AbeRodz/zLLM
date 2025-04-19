const cli = @import("./cli/cli.zig");
export var LLAMA_BUILD_NUMBER: c_int = 123;
export const LLAMA_BUILD_TARGET: [*:0]const u8 = "aarch-macOS";
export const LLAMA_COMMIT: [*:0]const u8 = "abc123";
export const LLAMA_COMPILER: [*:0]const u8 = "zig+clang";
pub fn main() !void {
    try cli.init();
}
