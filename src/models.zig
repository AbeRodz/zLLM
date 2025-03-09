const std = @import("std");

const HF_URL = "https://huggingface.co";
pub const CACHE_DIR = "/home/user/.cache/my_model_registry";

const ModelError = error{
    PrexistingModelFound,
    UnknownModel,
};

pub const ModelInfo = struct {
    name: []const u8,
    base_uri: []const u8,
    files: []const []const u8,
    cache_path: []const u8,

    const Self = @This();
    pub fn init(name: []const u8, base_uri: []const u8, files: []const []const u8) Self {
        return .{
            .name = name,
            .base_uri = base_uri,
            .files = files,
            .cache_path = comptime std.fmt.comptimePrint("{s}/{s}", .{ CACHE_DIR, name }),
        };
    }

    /// Returns the full local path of a file inside the cache directory
    pub fn localFilePath(self: Self, file_name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}", .{ self.cache_path, file_name });
    }
    /// Checks if all necessary files exist in the cache during RUNTIME
    pub fn isCached(self: Self) !bool {
        for (self.files) |file| {
            const path = self.localFilePath(file) catch return false;

            std.fs.cwd().access(path, .{}) catch return false;
        }
        return true;
    }
};

fn fmtBaseUri(model_name: []const u8) ![]const u8 {
    return comptime std.fmt.comptimePrint("{s}/{s}/resolve/main/", .{ HF_URL, model_name });
}

pub const MODELS = [_]ModelInfo{
    ModelInfo.init("vit-base", fmtBaseUri("google/vit-base-patch16-224") catch unreachable, &[_][]const u8{
        "model.safetensors?download=true",
        "config.json",
        "preprocessor_config.json",
    }),
    ModelInfo.init("clip-vit", fmtBaseUri("openai/clip-vit-base-patch32") catch unreachable, &[_][]const u8{
        "model.safetensors?download=true",
        "config.json",
        "preprocessor_config.json",
    }),
    ModelInfo.init("sam-vit-huge", fmtBaseUri("facebook/sam-vit-huge") catch unreachable, &[_][]const u8{
        "model.safetensors?download=true",
        "config.json",
    }),
};

pub fn findModel(name: []const u8) !?ModelInfo {
    for (MODELS) |m| {
        if (std.mem.eql(u8, m.name, name)) {
            if (try m.isCached()) {
                return error.PrexistingModelFound;
            }
            return m;
        }
    }
    return null;
}
