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

pub fn addGGMLSources(ctx: *BuildContext, compile: *CompileStep) void {
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
}

pub fn addMetalFrameworks(metal_compile: *CompileStep) void {
    for (llama_metal_frameworks) |src| {
        metal_compile.linkFramework(src);
    }
}
