const std = @import("std");
const CompileStep = std.Build.Step.Compile;
const BuildContext = @import("build_context.zig").BuildContext;

// C source files for llama.cpp
pub const llama_ggml_sources = &[_][]const u8{
    "ggml-alloc.c",
    "ggml-backend-reg.cpp",
    "ggml-backend.cpp",
    "ggml-opt.cpp",
    "ggml-quants.c",
    "ggml-threading.cpp",
    "ggml.c",
    "gguf.cpp",
};
pub const llama_ggml_cpu_sources = &[_][]const u8{
    "ggml-cpu-aarch64.cpp",
    "ggml-cpu.cpp",
    "ggml-cpu.c",
    "ops.cpp",
    "vec.cpp",
    "unary-ops.cpp",
    "binary-ops.cpp",
    "ggml-cpu-hbm.cpp",
    "ggml-cpu-quants.c",
    "ggml-cpu-traits.cpp",
    "amx/amx.cpp",
    "amx/mmq.cpp",
};

pub const llama_ggml_cuda_sources = &[_][]const u8{
    "ggml-cuda.cu",
    //"ggml-backend-impl.h",
    // "ggml-common.h",
};

pub fn globCuhFilesComptime(
    comptime dir_path: []const u8,
) []const []const u8 {
    const fs = std.fs;

    var files: [60][]const u8 = undefined; // fixed max capacity

    var count: usize = 0;

    var dir = fs.cwd().openDir(dir_path, .{ .access_sub_paths = false, .iterate = true }) catch @panic("failed to open dir");
    //defer (@as(*fs.Dir, @ptrCast(dir))).close();
    defer dir.close();

    var it = dir.iterate();

    while (true) {
        const entry = it.next() catch unreachable;
        if (entry == null) break;

        if (entry.?.kind == .file and std.mem.endsWith(u8, entry.?.name, ".cuh")) {
            files[count] = entry.?.name;
            // std.debug.print("src: {s}\n", .{files[count]});
            count += 1;
            if (count >= files.len) break; // prevent overflow
        }
    }

    return files[0..count];
}

fn collectCuhFiles(b: *std.Build, allocator: std.mem.Allocator, dir_path: []const u8) !std.ArrayList(std.Build.LazyPath) {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .access_sub_paths = false, .iterate = true });
    defer dir.close();

    var list = std.ArrayList(std.Build.LazyPath).init(allocator);
    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.basename, ".cuh")) {
            const full_path = try std.fs.path.join(allocator, &.{ dir_path, entry.path });
            const lazy = b.path(full_path);
            try list.append(lazy);
        }
    }

    return list;
}
pub const llama_ggml_metal_sources = &[_][]const u8{
    "ggml-metal",
};
pub const llama_cpp_sources = &[_][]const u8{
    "llama-adapter.cpp",
    "llama-io.cpp",
    "llama-cparams.cpp",
    "llama-graph.cpp",
    "llama-io.cpp",
    "llama-arch.cpp",
    "llama-batch.cpp",
    "llama-chat.cpp",
    "llama-context.cpp",
    "llama-grammar.cpp",
    "llama-hparams.cpp",
    "llama-impl.cpp",
    "llama-kv-cache.cpp",
    "llama-mmap.cpp",
    "llama-model-loader.cpp",
    "llama-model.cpp",
    "llama-sampling.cpp",
    "llama-vocab.cpp",
    "llama.cpp",
    "unicode-data.cpp",
    "unicode.cpp",
};

pub const llama_common_sources = &[_][]const u8{
    "common.cpp",
    "sampling.cpp",
    "console.cpp",

    "json-schema-to-grammar.cpp",
    "speculative.cpp",
    "ngram-cache.cpp",

    "log.cpp",
    "arg.cpp",
};

pub const include_paths = &[_][]const u8{
    "llama.cpp",
    "llama.cpp/include",
    "llama.cpp/common",
    "llama.cpp/ggml/include",
    "llama.cpp/ggml/src",
    "llama.cpp/ggml/src/ggml-cpu",
    "llama.cpp/ggml/src/ggml-blas",
};

pub const include_metal_paths = &[_][]const u8{
    "include",
    "src",
    "src/ggml-metal",
};

pub const llama_metal_frameworks = &[_][]const u8{
    "Accelerate",
    "Foundation",
    "AppKit",
    "Metal",
    "MetalKit",
};

pub fn addCBuildSources(ctx: *BuildContext, b: *std.Build) void {
    ctx.common(ctx.lib);
    for (include_paths) |path| {
        ctx.lib.addIncludePath(b.path(path));
    }

    for (llama_cpp_sources) |src| {
        ctx.lib.addCSourceFile(.{
            .file = b.path(src),
            .flags = &[_][]const u8{
                "-g",
                "-O3",
                "-pthread",
                "-std=c++17",
            },
        });
    }
}

pub fn addMetalIncludes(ctx: *BuildContext, compile: *CompileStep) void {
    for (include_metal_paths) |path| {
        compile.addIncludePath(ctx.path(&.{ "ggml", path }));
    }
}
pub fn addLLama(ctx: *BuildContext, compile: *CompileStep) void {
    ctx.common(compile);
    compile.addIncludePath(ctx.path(&.{"include"}));
    for (llama_cpp_sources) |src| {
        compile.addCSourceFile(.{
            .file = ctx.srcPath(src),
            .flags = ctx.flags(),
        });
    }

    for (llama_common_sources) |src| {
        compile.addCSourceFile(.{
            .file = ctx.path(&.{ "common", src }),
            .flags = ctx.flags(),
        });
    }
}

pub fn addGGMLSources(ctx: *BuildContext, compile: *CompileStep) !void {
    for (llama_ggml_sources) |src| {
        compile.addCSourceFile(.{
            .file = ctx.path(&.{ "ggml", "src", src }),
            .flags = ctx.flags(),
        });
    }
    for (llama_ggml_cpu_sources) |src| {
        compile.addCSourceFile(.{
            .file = ctx.path(&.{ "ggml", "src", "ggml-cpu", src }),
            .flags = ctx.flags(),
        });
    }
    if (ctx.platform == .Cuda) {
        const cuhs = try collectCuhFiles(ctx.build, ctx.build.allocator, "llama.cpp/ggml/src/ggml-cuda");
        for (cuhs.items) |src| {
            //std.debug.print("cuhs: {s}\n", .{src});
            compile.addIncludePath(src);
        }
        // for (llama_ggml_cuda_sources) |src| {
        //     compile.addCSourceFile(.{
        //         .file = ctx.path(&.{ "ggml", "src", "ggml-cuda", src }),
        //         .flags = ctx.flags(),
        //     });
        // }
    }
}

pub fn addMetalFrameworks(metal_compile: *CompileStep) void {
    for (llama_metal_frameworks) |src| {
        metal_compile.linkFramework(src);
    }
}
