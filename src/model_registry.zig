const std = @import("std");
const ModelInfo = @import("models.zig").ModelInfo;

pub const CACHE_DIR = "/home/user/.cache/my_model_registry"; // Change for cross-platform support

pub const ModelRegistry = struct {
    model_info: ModelInfo,
    cache_path: []const u8,

    const Self = @This();

    pub fn init(model_info: ModelInfo) Self {
        return .{
            .model_info = model_info,
            .cache_path = std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}", .{ CACHE_DIR, model_info.name }) catch unreachable,
        };
    }

    /// Returns the full local path of a file inside the cache directory
    pub fn localFilePath(self: Self, file_name: []const u8) []const u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}", .{ self.cache_path, file_name }) catch unreachable;
    }

    /// Checks if all necessary files exist in the cache
    pub fn isCached(self: Self) bool {
        for (self.files) |file| {
            if (std.fs.cwd().access(self.localFilePath(file), .{}) != .{}) {
                return false; // If any file is missing, return false
            }
        }
        return true;
    }
};
// TODO
pub const Registered = [_]ModelRegistry{
    ModelRegistry.init(),
    ModelInfo.init("vit-base", "https://huggingface.co/google/vit-base-patch16-224/resolve/main/", &[_][]const u8{
        "model.safetensors?download=true",
        "config.json",
        "preprocessor_config.json",
    }),
    ModelInfo.init("clip-vit", "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/", &[_][]const u8{
        "model.safetensors?download=true",
        "config.json",
        "preprocessor_config.json",
    }),
    ModelInfo.init("sam-vit-huge", "https://huggingface.co/facebook/sam-vit-huge/resolve/main/", &[_][]const u8{
        "model.safetensors?download=true",
        "config.json",
    }),
};
