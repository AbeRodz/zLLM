const std = @import("std");
const CompileStep = std.Build.Step.Compile;
const Module = std.Build.Module;
const BuildContext = @import("build_context.zig").BuildContext;
const CompilerConfig = @import("build_platform.zig").CompilerConfig;
const Platform = @import("build_platform.zig").Platform;
const OsToPlatform = @import("build_platform.zig").OsToPlatform;
const addCBuildSources = @import("build_csources.zig").addCBuildSources;
const tokamak = @import("tokamak");

pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
    source_path: []const u8 = "",
    platform: Platform,
};
pub const Context = struct {
    const Self = @This();
    b: *std.Build,
    options: Options,
    llama: BuildContext,
    module: *Module,
    llama_h_module: *Module,
    ggml_h_module: *Module,

    pub fn init(b: *std.Build, options: Options) Self {
        var llama_cpp = BuildContext.init(
            b,
            options.target,
            options.platform,
            options.optimize,
        );
        const llama_h_module = llama_cpp.moduleLlama();
        const ggml_h_module = llama_cpp.moduleGgml();
        const imports: []const std.Build.Module.Import = &.{
            .{
                .name = "llama.h",
                .module = llama_h_module,
            },
            .{
                .name = "ggml.h",
                .module = ggml_h_module,
            },
        };
        const mod = b.createModule(.{
            .root_source_file = b.path(b.pathJoin(&.{ options.source_path, "src/llama/llama.zig" })),
            .imports = imports,
        });

        return .{
            .b = b,
            .options = options,
            .llama = llama_cpp,
            .module = mod,
            .llama_h_module = llama_h_module,
            .ggml_h_module = ggml_h_module,
        };
    }
    pub fn link(self: *Self, comp: *CompileStep) void {
        self.llama.link(comp);
    }
};
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const platform = OsToPlatform(target, b);
    std.debug.print("Target OS: {}, platform: {}\n", .{ target.result.os, platform });
    var ctx = Context.init(b, .{
        .platform = platform,
        .source_path = "",
        .target = target,
        .optimize = optimize,
    });
    // Install the library and executable

    const exe = ctx.b.addExecutable(.{
        .name = "zLLM",
        .root_source_file = b.path("src/main.zig"),
        .target = ctx.options.target,
        .optimize = ctx.options.optimize,
    });
    exe.addIncludePath(ctx.llama.path(&.{"include"}));
    exe.addIncludePath(ctx.llama.path(&.{"common"}));
    exe.addIncludePath(ctx.llama.path(&.{ "ggml", "include" }));
    exe.addIncludePath(ctx.llama.path(&.{ "ggml", "src" }));
    exe.want_lto = false;
    ctx.llama.common(exe);
    ctx.link(exe);

    const uuid = b.dependency("uuid", .{
        .target = ctx.options.target,
        .optimize = ctx.options.optimize,
    });
    exe.root_module.addImport("uuid", uuid.module("uuid"));
    tokamak.setup(exe, .{});

    ctx.b.installArtifact(exe);
    // Run step
    const run_cmd = ctx.b.addRunArtifact(exe);
    run_cmd.step.dependOn(ctx.b.getInstallStep());
    if (ctx.b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = ctx.b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const lib_unit_tests = ctx.b.addTest(.{
        .root_source_file = ctx.b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = ctx.b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = ctx.b.addTest(.{
        .root_source_file = ctx.b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = ctx.b.addRunArtifact(exe_unit_tests);

    const test_step = ctx.b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
