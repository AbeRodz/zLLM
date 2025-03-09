const std = @import("std");

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
};

pub const MODELS = [_]ModelInfo{
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

pub fn findModel(name: []const u8) ?ModelInfo {
    for (MODELS) |m| {
        if (std.mem.eql(u8, m.name, name)) {
            return m;
        }
    }
    return null;
}
