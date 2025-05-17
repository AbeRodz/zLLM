const std = @import("std");
const LazyPath = std.Build.LazyPath;
const CompileStep = std.Build.Step.Compile;
const Platform = @import("build_platform.zig").Platform;
const Csources = @import("build_csources.zig");
const MetalContext = @import("build_metal.zig").MetalContext;

pub const BuildContext = struct {
    path_prefix: []const u8 = "",
    build: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
    platform: Platform,
    build_info: *CompileStep,
    lib: ?*CompileStep = null,

    const Self = @This();
    pub fn init(
        b: *std.Build,
        target: std.Build.ResolvedTarget,
        platform: Platform,
        optimize: std.builtin.Mode,
    ) Self {
        const zig_version = @import("builtin").zig_version_string;
        const path_prefix = b.pathJoin(&.{ thisPath(), "/llama.cpp" });
        const build_info_path = b.pathJoin(&.{ "common", "build-info.zig" });
        const build_info = b.fmt(
            \\pub export var LLAMA_BUILD_NUMBER : c_int = {};
            \\pub export var LLAMA_COMMIT = "{s}";
            \\pub export var LLAMA_COMPILER = "Zig {s}";
            \\pub export var LLAMA_BUILD_TARGET = "{s}_{s}";
            \\
        , .{ 0, "1230132", zig_version, target.result.zigTriple(b.allocator) catch unreachable, @tagName(optimize) });

        return Self{
            .path_prefix = path_prefix,
            .build = b,
            .target = target,
            .optimize = optimize,
            .platform = platform,
            .build_info = b.addObject(.{ .name = "llama-build-info", .target = target, .optimize = optimize, .root_source_file = b.addWriteFiles().add(build_info_path, build_info) }),
        };
    }
    /// just builds everything needed and links it to your target
    pub fn link(ctx: *Self, comp: *CompileStep) !void {
        const lib = try ctx.library();
        comp.linkLibrary(lib);
    }

    /// build single library containing everything
    pub fn library(ctx: *Self) !*CompileStep {
        if (ctx.lib) |l| return l;
        const lib = ctx.build.addStaticLibrary(std.Build.StaticLibraryOptions{
            .name = "llama.cpp",
            .target = ctx.target,
            .optimize = ctx.optimize,
        });
        lib.root_module.addCMacro("LOG_DISABLE_LOGS", "1");
        lib.root_module.addCMacro("GGML_USE_CPU", "1");
        lib.root_module.addCMacro("LLAMA_FATAL_WARNINGS", "ON");
        if (ctx.platform == .Metal) {
            lib.root_module.addCMacro("GGML_USE_METAL", "");
            lib.root_module.addCMacro("GGML_METAL_USE_BF16", "ON");
            lib.root_module.addCMacro("GGML_METAL_EMBED_LIBRARY", "ON");
        }
        if (ctx.platform == .Cuda) {
            lib.root_module.addCMacro("DGGML_USE_CUDA", "");
            lib.root_module.addCMacro("GGML_CUDA_USE_GRAPHS", "ON");
        }
        try ctx.addAll(lib);
        if (ctx.target.result.abi != .msvc)
            lib.root_module.addCMacro("_GNU_SOURCE", "");
        ctx.lib = lib;
        return lib;
    }
    pub fn addAll(ctx: *Self, compile: *CompileStep) !void {
        ctx.addBuildInfo(compile);
        try ctx.addGgml(compile);
        Csources.addLLama(ctx, compile);
    }
    /// zig module with translated headers
    pub fn moduleLlama(ctx: *Self) *std.Build.Module {
        const tc = ctx.build.addTranslateC(.{
            .root_source_file = ctx.includePath("llama.h"),
            .target = ctx.target,
            .optimize = ctx.optimize,
        });
        tc.addSystemIncludePath(ctx.path(&.{ "ggml", "include" }));
        tcDefineCMacro(tc, "NDEBUG", null); // otherwise zig is unhappy about c ASSERT macro
        return tc.addModule("llama.h");
    }

    /// zig module with translated headers
    pub fn moduleGgml(ctx: *Self) *std.Build.Module {
        const tc = ctx.build.addTranslateC(.{
            .root_source_file = ctx.path(&.{ "ggml", "include", "ggml.h" }),
            .target = ctx.target,
            .optimize = ctx.optimize,
        });

        tcDefineCMacro(tc, "LLAMA_SHARED", null);
        tcDefineCMacro(tc, "NDEBUG", null);

        return tc.addModule("ggml.h");
    }

    pub fn addBuildInfo(ctx: *Self, compile: *CompileStep) void {
        compile.addObject(ctx.build_info);
    }
    pub fn addGgml(ctx: *Self, compile: *CompileStep) !void {
        ctx.common(compile);
        compile.addIncludePath(ctx.path(&.{ "ggml", "include" }));
        compile.addIncludePath(ctx.path(&.{ "ggml", "src" }));

        if (ctx.platform == .Metal) {
            compile.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-metal" }));
        }
        if (ctx.platform == .Cuda) {
            compile.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-cuda" }));
        }

        compile.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-cpu" }));

        compile.addIncludePath(ctx.path(&.{ "ggml", "src", "ggml-blas" }));
        compile.linkLibCpp();

        const common_src = ctx.path(&.{ "ggml", "src", "ggml-common.h" });
        const common_dst = "ggml-common.h";
        const common_install_step = ctx.build.addInstallFile(common_src, common_dst);

        if (ctx.platform == .Metal) {
            const metallib_compile = ctx.build.addSystemCommand(&.{
                "xcrun", "-sdk", "macosx", "metallib", ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "ggml-metal.air" }), "-o", ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "default.metallib" }),
            });
            const metal_ctx = MetalContext.metalLibrary(ctx, common_install_step);
            metallib_compile.step.dependOn(&metal_ctx.compile.step);
            // Install the metal shader source file to bin directory
            const metal_shader_install = ctx.build.addInstallBinFile(ctx.path(&.{ "ggml", "src", "ggml-metal", "ggml-metal.metal" }), "ggml-metal.metal");
            const default_lib_install = ctx.build.addInstallBinFile(.{ .cwd_relative = ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "default.metallib" }) }, "default.metallib");
            metal_shader_install.step.dependOn(&metallib_compile.step);
            default_lib_install.step.dependOn(&metal_shader_install.step);
            // Link the metal library with the main compilation
            compile.linkLibrary(metal_ctx.lib);
            compile.step.dependOn(&metal_ctx.lib.step);
            compile.step.dependOn(&default_lib_install.step);
        }

        if (ctx.platform == .Cuda) {}
        // if (ctx.platform == .Cuda){

        // }

        try Csources.addGGMLSources(ctx, compile);
    }

    pub fn flags(ctx: Self) []const []const u8 {
        _ = ctx;
        return &.{"-fno-sanitize=undefined"};
    }
    pub fn common(self: Self, lib: *CompileStep) void {
        lib.linkSystemLibrary("stdc++");
        lib.linkSystemLibrary("pthread");
        lib.linkSystemLibrary("m");
        lib.linkLibCpp();
        lib.addIncludePath(self.build.path("llama.cpp/common"));
    }
    pub fn path(self: Self, subpath: []const []const u8) LazyPath {
        const sp = self.build.pathJoin(subpath);
        return .{ .cwd_relative = self.build.pathJoin(&.{ self.path_prefix, sp }) };
    }
    pub fn srcPath(self: Self, p: []const u8) LazyPath {
        return .{ .cwd_relative = self.build.pathJoin(&.{ self.path_prefix, "src", p }) };
    }
    pub fn includePath(self: Self, p: []const u8) LazyPath {
        return .{ .cwd_relative = self.build.pathJoin(&.{ self.path_prefix, "include", p }) };
    }
};

fn thisPath() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}

fn tcDefineCMacro(tc: *std.Build.Step.TranslateC, comptime name: []const u8, comptime value: ?[]const u8) void {
    tc.defineCMacroRaw(name ++ "=" ++ (value orelse "1"));
}

pub fn addDefines(comp: *CompileStep) void {
    //if (self.cuda) comp.root_module.addCMacro("GGML_USE_CUDA", "");
    comp.root_module.addCMacro("GGML_USE_METAL", "");
    //if (self.cpu) comp.root_module.addCMacro("GGML_USE_CPU", "");
}
