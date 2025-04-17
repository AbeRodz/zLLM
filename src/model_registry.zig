const std = @import("std");

const HF_URL = "https://huggingface.co";

const ModelError = error{
    PreexistingModelFound,
    UnknownModel,
};

pub const ModelInfo = struct {
    name: []const u8,
    base_uri: []const u8,
    files: []const []const u8,

    const Self = @This();
    pub fn init(name: []const u8, base_uri: []const u8, files: []const []const u8) Self {
        return .{
            .name = name,
            .base_uri = base_uri,
            .files = files,
        };
    }

    /// Returns the full local path of a file inside the cache directory
    pub fn localFilePath(_: Self, model_name: []const u8, file_name: []const u8) ![]const u8 {
        const cache_dir = try getCacheDir(std.heap.page_allocator);
        return std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}/{s}", .{ cache_dir, model_name, file_name });
    }
    /// Checks if all necessary files exist in the cache during RUNTIME
    pub fn isCached(self: Self) !bool {
        for (self.files) |file| {
            const path = self.localFilePath(self.name, file) catch return false;

            std.fs.cwd().access(path, .{}) catch return false;
        }
        return true;
    }
};

fn fmtBaseUri(model_name: []const u8) ![]const u8 {
    return comptime std.fmt.comptimePrint("{s}/{s}/resolve/main/", .{ HF_URL, model_name });
}

pub fn getCacheDir(allocator: std.mem.Allocator) ![]u8 {
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch return error.MissingHome;
    defer allocator.free(home);

    return try std.fs.path.join(allocator, &.{ home, ".cache", "zLLM" });
}

pub fn checkDir(allocator: std.mem.Allocator) !void {
    const cache_dir = try getCacheDir(allocator);
    defer allocator.free(cache_dir);

    std.fs.makeDirAbsolute(cache_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };

    var dir = try std.fs.openDirAbsolute(cache_dir, .{});
    defer dir.close();
}

pub const MODELS = [_]ModelInfo{
    ModelInfo.init("vit-base", fmtBaseUri("google/vit-base-patch16-224") catch unreachable, &[_][]const u8{
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
    }),
    ModelInfo.init("clip-vit", fmtBaseUri("openai/clip-vit-base-patch32") catch unreachable, &[_][]const u8{
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
    }),
    ModelInfo.init("sam-vit-huge", fmtBaseUri("facebook/sam-vit-huge") catch unreachable, &[_][]const u8{
        "model.safetensors",
        "config.json",
    }),
    ModelInfo.init("gemma-3-1b", fmtBaseUri("google/gemma-3-1b-it") catch unreachable, &[_][]const u8{
        "tokenizer.json",
        "tokenizer.model",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "added_tokens.json",
        "model.safetensors",
    }),
    ModelInfo.init("gpt-2", fmtBaseUri("openai-community/gpt2") catch unreachable, &[_][]const u8{
        "model.safetensors",
        "config.json",
        "tokenizer.json",
    }),
};

pub fn findModel(name: []const u8) !?ModelInfo {
    for (MODELS) |m| {
        if (std.mem.eql(u8, m.name, name)) {
            if (try m.isCached()) {
                return error.PreexistingModelFound;
            }
            return m;
        }
    }
    return null;
}

//temporary
pub fn findModelErrorless(name: []const u8) !?ModelInfo {
    for (MODELS) |m| {
        if (std.mem.eql(u8, m.name, name)) {
            return m;
        }
    }
    return null;
}
pub fn listAvailableModels(allocator: std.mem.Allocator) ![]ModelInfo {
    var list = std.ArrayList(ModelInfo).init(allocator);

    for (MODELS) |m| {
        if (try m.isCached()) {
            try list.append(m);
        }
    }
    return list.toOwnedSlice();
}
