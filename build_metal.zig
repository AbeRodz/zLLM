const std = @import("std");
const Step = std.Build.Step;
const CompileStep = Step.Compile;

const BuildContext = @import("build_context.zig").BuildContext;
const Csources = @import("build_csources.zig");

pub const MetalContext = struct {
    lib: *CompileStep,
    compile: *Step.Run,

    // pub fn xcrunCompile(ctx: BuildContext) !void {
    //     const metallib_compile = ctx.build.addSystemCommand(&.{
    //         "xcrun", "-sdk", "macosx", "metallib", ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "ggml-metal.air" }), "-o", ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "default.metallib" }),
    //     });
    // }

    pub fn metalLibrary(ctx: *BuildContext, common: *Step.InstallFile) MetalContext {
        const metal_lib = ctx.build.addStaticLibrary(.{
            .name = "ggml-metal",
            .target = ctx.target,
            .optimize = ctx.optimize,
        });

        Csources.addMetalIncludes(ctx, metal_lib);
        Csources.addMetalFrameworks(metal_lib);

        metal_lib.addCSourceFile(.{ .file = ctx.path(&.{ "ggml", "src", "ggml-metal", "ggml-metal.m" }), .flags = ctx.flags() });
        const metal_files = [_][]const u8{
            "ggml-metal.metal",
            "ggml-metal-impl.h",
        };
        // Compile the metal shader [requires xcode installed]
        const metal_compile = ctx.build.addSystemCommand(&.{
            "xcrun",          "-sdk",                                                                      "macosx", "metal",
            "-fno-fast-math", "-g",                                                                        "-c",     ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "ggml-metal.metal" }),
            "-o",             ctx.build.pathJoin(&.{ ctx.build.install_path, "metal", "ggml-metal.air" }), "-I",     "llama.cpp/ggml/src",
        });
        metal_compile.step.dependOn(&common.step);
        for (metal_files) |file| {
            const src = ctx.path(&.{ "ggml", "src", "ggml-metal", file });
            const dst = ctx.build.pathJoin(&.{ "metal", file });
            const install_step = ctx.build.addInstallFile(src, dst);
            metal_compile.step.dependOn(&install_step.step);
        }
        return MetalContext{
            .lib = metal_lib,
            .compile = metal_compile,
        };
    }
};
