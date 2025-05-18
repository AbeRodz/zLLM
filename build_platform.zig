const std = @import("std");
// Supported platforms
pub const Platform = enum {
    Metal,
    Cuda,
    Cpu,
};

pub fn OsToPlatform(target: std.Build.ResolvedTarget, build: *std.Build) Platform {
    const os = target.result.os.tag;
    const use_cuda = build.option(bool, "use-cuda", "Use CUDA backend") orelse false;
    return switch (os) {
        .macos => Platform.Metal,
        .linux => if (use_cuda) Platform.Cuda else Platform.Cpu,
        else => Platform.Cpu,
    };
}

pub const CompilerConfig = struct {
    platform: Platform,

    const Self = @This();
    // Method to get the compile flags for the given platform
    pub fn get_compile_flags(self: Self) []const u8 {
        return switch (self.platform) {
            .Metal => &[_][]const u8{ "-DGGML_METAL", "-fno-objc-arc", "-ObjC" },
            .Cuda => &[_][]const u8{"-DGGML_CUDA"},
            .Cpu => &[_][]const u8{},
        };
    }
    pub fn addMacros(self: Self, lib: *std.Build.Step.Compile) void {
        switch (self.platform) {
            .Metal => lib.root_module.addCMacro("GGML_USE_METAL", ""),
            .Cuda => lib.root_module.addCMacro("GGML_USE_CUDA", ""),
            .Cpu => lib.root_module.addCMacro("GGML_USE_CPU", ""),
        }
    }
    // Method to link platform-specific frameworks or libraries
    pub fn link_platform(self: Self, exe: *std.Build.Step.Compile) void {
        const metal_frameworks = &[_][]const u8{
            "Foundation",
            //"QuartzCore",
            //"AppKit",
            "Metal",
            "MetalKit",
        };

        const cuda_libraries = &[_][]const u8{
            "cuda",
        };

        switch (self.platform) {
            .Metal => {
                for (metal_frameworks) |framework| {
                    exe.linkFramework(framework);
                }
            },
            .Cuda => {
                for (cuda_libraries) |library| {
                    exe.linkSystemLibrary(library);
                }
            },
            .Cpu => {},
        }
    }
};
