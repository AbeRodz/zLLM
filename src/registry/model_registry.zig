const std = @import("std");
const json = std.json;
const embedded_manifest = @embedFile("registry_manifest.json");

const ModelError = error{
    PreexistingModelFound,
    UnknownModel,
    MissingHome,
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
    pub fn isGGUFCached(self: Self) !bool {
        const path = self.localFilePath(self.name, "model.gguf") catch return false;
        std.fs.cwd().access(path, .{}) catch return false;
        return true;
    }
    pub fn loadGGUFModelBuffer(self: ModelInfo, allocator: std.mem.Allocator) ![]u8 {
        if (self.files.len == 0) return error.UnknownModel;
        const path = try self.localFilePath(self.name, "model.gguf");
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        const file_size = try file.getEndPos();
        const buffer = try allocator.alloc(u8, file_size);

        _ = try file.readAll(buffer);

        return buffer;
    }

    pub fn loadSafetensorsBuffer(self: ModelInfo, allocator: std.mem.Allocator) ![]u8 {
        if (self.files.len == 0) return error.UnknownModel;

        const path = try self.localFilePath(self.name, "model.safetensors");
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        const file_size = try file.getEndPos();
        const buffer = try allocator.alloc(u8, file_size);
        _ = try file.readAll(buffer);

        return buffer;
    }
    // TODO
    pub fn SeekSafetensorsBuffer(self: ModelInfo, allocator: std.mem.Allocator) ![]u8 {
        if (self.files.len == 0) return error.UnknownModel;

        const path = try self.localFilePath(self.name, "model.safetensors");
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        const file_size = try file.getEndPos();
        const buffer = try allocator.alloc(u8, file_size);
        _ = try file.readAll(buffer);

        return buffer;
    }
};

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

pub fn parseModels(allocator: std.mem.Allocator) ![]ModelInfo {
    const parsed = try json.parseFromSlice([]ModelInfo, allocator, embedded_manifest, .{
        .allocate = .alloc_always,
    });
    return parsed.value;
}

pub fn findModel(name: []const u8) !?ModelInfo {
    const models = try parseModels(std.heap.page_allocator);
    for (models) |m| {
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
    const models = try parseModels(std.heap.page_allocator);
    for (models) |m| {
        if (std.mem.eql(u8, m.name, name)) {
            return m;
        }
    }
    return null;
}
pub fn listAvailableModels(allocator: std.mem.Allocator) ![]ModelInfo {
    const models = try parseModels(allocator);
    defer allocator.free(models);
    var list = std.ArrayList(ModelInfo).init(allocator);

    for (models) |m| {
        if (try m.isCached()) {
            try list.append(m);
        }
    }
    return list.toOwnedSlice();
}
