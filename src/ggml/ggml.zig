const std = @import("std");
const Tensors = @import("tensors.zig").Tensors;
const KV = @import("KV.zig").KVMap;
const ContainerGGUF = @import("gguf.zig").ContainerGGUF;
const FileReader = @import("reader.zig").FileReader;
// Magic constant for `ggml` files (unversioned).
const FILE_MAGIC_GGML = 0x67676d6c;
// Magic constant for `ggml` files (versioned, ggmf).
const FILE_MAGIC_GGMF = 0x67676d66;
// Magic constant for `ggml` files (versioned, ggjt).
const FILE_MAGIC_GGJT = 0x67676a74;
// Magic constant for `ggla` files (LoRA adapter).
const FILE_MAGIC_GGLA = 0x67676C61;
// Magic constant for `gguf` files (versioned, gguf)
const FILE_MAGIC_GGUF_LE = 0x46554747;
const FILE_MAGIC_GGUF_BE = 0x47475546;

pub fn detectContentType(b: []const u8) []const u8 {
    if (b.len < 4) return "";

    const magic = std.mem.readInt(u32, b[0..4], .little);

    return switch (magic) {
        FILE_MAGIC_GGML => "ggml",
        FILE_MAGIC_GGMF => "ggmf",
        FILE_MAGIC_GGJT => "ggjt",
        FILE_MAGIC_GGLA => "ggla",
        FILE_MAGIC_GGUF_LE, FILE_MAGIC_GGUF_BE => "gguf",
        else => "",
    };
}
pub const GGUFDescriptor = struct {
    version: u32,
    alignment: u32,
    kv_pairs: u64,
    tensors: u64,
    parameter_count: ?u64 = null,
};
pub fn printDescriptor(desc: GGUFDescriptor) void {
    std.debug.print(
        \\
        \\Model GGUF Description
        \\----------------
        \\Version          : {d}
        \\KV Pairs         : {d}
        \\Tensors          : {d}
        \\Alignment        : {d}
        \\Parameters       : {?}
        \\
    , .{
        desc.version,
        desc.kv_pairs,
        desc.tensors,
        desc.alignment,
        desc.parameter_count,
    });
}
pub const GGML = struct {
    Name: []const u8,
    KV: KV,
    Tensors: Tensors,
    Length: i64,

    const Self = @This();
    pub fn decode(buffer: []const u8, max_array_size: usize, allocator: std.mem.Allocator) !GGML {
        var reader = FileReader.init(buffer);

        // Read magic number (first 4 bytes)
        const magic_bytes = try reader.readBytes(4);
        const magic = std.mem.readVarInt(u32, magic_bytes, .little);
        const size = @sizeOf(ContainerGGUF);
        const alignment = @alignOf(ContainerGGUF);

        std.debug.print("ContainerGGUF size: {} bytes\n", .{size});
        std.debug.print("ContainerGGUF alignment: {} bytes\n", .{alignment});
        // Pick endian and initialize GGUF container
        var container: ContainerGGUF = switch (magic) {
            FILE_MAGIC_GGUF_LE => ContainerGGUF.init(.little, max_array_size),
            FILE_MAGIC_GGUF_BE => ContainerGGUF.init(.big, max_array_size),
            else => return error.InvalidFileMagic,
        };

        // Decode model with FileReader
        const model = try container.decode(&reader, allocator);

        const ggml = GGML{
            .Name = model.kv.architecture(),
            .KV = model.kv,
            .Tensors = model.tensors,
            .Length = @as(i64, @intCast(buffer.len)),
        };

        return ggml;
    }
    pub fn describeGGUF(buffer: []const u8, max_array_size: usize, allocator: std.mem.Allocator) !GGUFDescriptor {
        var reader = FileReader.init(buffer);

        const magic_bytes = try reader.readBytes(4);
        const magic = std.mem.readVarInt(u32, magic_bytes, .little);

        var container: ContainerGGUF = switch (magic) {
            FILE_MAGIC_GGUF_LE => ContainerGGUF.init(.little, max_array_size),
            FILE_MAGIC_GGUF_BE => ContainerGGUF.init(.big, max_array_size),
            else => return error.InvalidFileMagic,
        };

        var model = try container.decode(&reader, allocator);

        const descriptor = GGUFDescriptor{
            .version = container.version,
            .alignment = model.alignment.?,
            .kv_pairs = model.numKV(),
            .parameter_count = model.parameters,
            .tensors = model.numTensor(),
        };

        return descriptor;
    }

    pub fn supportsKVCacheType(_: Self, cache_type: []const u8) bool {
        return std.mem.eql(u8, cache_type, "f16") or
            std.mem.eql(u8, cache_type, "q8_0") or
            std.mem.eql(u8, cache_type, "q4_0");
    }

    pub fn GraphSize(self: Self, context: u64, batch: u64, num_parallel: u32, kv_cache_type: []const u8) !struct {
        kv: []u64,
        partial_offload: u64,
        full_offload: u64,
    } {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        const alloc = gpa.allocator();

        const kvmap = self.KV;

        const embedding = kvmap.EmbeddingLength();
        const heads = kvmap.HeadCount();
        const heads_kv = kvmap.HeadCountKV();

        // vocab length from array of strings
        const tokenizer_tokens = kvmap.KV.get("tokenizer.ggml.tokens") orelse return error.MissingTokenizerTokens;
        const vocab = @as(u64, @intCast(tokenizer_tokens.strs.len));

        const embedding_heads = kvmap.EmbeddingHeadCount();
        const embedding_heads_k = kvmap.EmbeddingHeadCountK();
        const embedding_heads_v = kvmap.EmbeddingHeadCountV();

        const layers = try self.Tensors.groupLayers(alloc);

        const bytes_per_element = kvCacheBytesPerElement(kv_cache_type);

        const block_count = kvmap.BlockCount();
        const kv = try alloc.alloc(u64, block_count);
        for (kv) |*v| {
            v.* = @as(u64, @intFromFloat(@as(f64, context) * @as(f64, @floatFromInt(embedding_heads_k + embedding_heads_v)) * @as(f64, @floatFromInt(heads_kv)) * bytes_per_element));
        }

        const arch = kvmap.architecture();
        var partial_offload: u64 = 0;
        var full_offload: u64 = 0;

        if (std.mem.eql(u8, arch, "llama") or std.mem.eql(u8, arch, "llama4")) {
            full_offload = @max(
                4 * batch * (1 + 4 * embedding + context * (1 + heads)),
                4 * batch * (embedding + vocab),
            );

            partial_offload = 4 * batch * embedding;
            partial_offload += @max(
                4 * batch * (1 + embedding + @max(context, embedding)) +
                    embedding * embedding * 9 / 16 +
                    4 * context * (batch * heads + embedding_heads * heads_kv),
                4 * batch * (embedding + vocab) + embedding * vocab * 105 / 128,
            );

            // mixtral 8x22b
            if (layers.get("blk.0")) |blk0| {
                if (blk0.get("ffn_gate_exps.weight")) |_| {
                    const ff = kvmap.Uint("feed_forward_length");
                    partial_offload = @max(
                        3 * blk0.get("ffn_gate_exps.weight").?.Size() +
                            4 * batch * (2 * ff + heads_kv + embedding + context + embedding_heads * heads_kv),
                        4 * (context * batch * heads +
                            context * embedding_heads * heads_kv +
                            batch * 1024 +
                            embedding_heads * heads_kv * batch),
                    );
                } else if (blk0.get("ffn_gate.0.weight")) |ffn_gate_weight| {
                    const ffn_gate_weight1 = ffn_gate_weight.Shape[1];
                    full_offload = 4 * batch * (2 + 3 * embedding + context * (1 + heads) + 2 * heads_kv + ffn_gate_weight1);
                    partial_offload = @max(
                        4 * batch * (3 + embedding_heads * heads_kv + embedding + context * (1 + heads) + ffn_gate_weight1) +
                            (embedding * embedding + 3 * embedding * heads_kv * ffn_gate_weight1) * 9 / 16,
                        4 * batch * (1 + 2 * embedding + context * (1 + heads)) +
                            embedding * (6 * context * heads_kv / heads + embedding * 9 / 16),
                    );
                }
            }

            if (std.mem.eql(u8, arch, "gemma") or std.mem.eql(u8, arch, "gemma2") or std.mem.eql(u8, arch, "gemma3")) {
                full_offload = @max(
                    4 * batch * (embedding + vocab),
                    4 * batch * (2 + context + context * heads + 2 * embedding + 2 * embedding_heads_k * heads),
                );

                partial_offload = @max(
                    4 * embedding * batch + embedding * vocab * 105 / 128 + 4 * vocab * batch,
                    4 * batch * (2 * embedding + 1 + 2 * embedding_heads_k * heads + context + context * heads) +
                        4 * embedding_heads_k * context * 8 +
                        embedding * embedding_heads_k * heads * 9 / 16,
                );

                if (std.mem.eql(u8, arch, "gemma3")) {
                    const gemma3_global_cache_count = 6;
                    const sliding_window = @as(u64, @intCast(num_parallel)) * kvmap.Uint("attention.sliding_window") + batch;

                    for (kv, 0..) |*v, i| {
                        if ((i + 1) % gemma3_global_cache_count != 0) {
                            v.* = @as(u64, @as(f64, @intFromFloat(sliding_window)) *
                                @as(f64, embedding_heads_k + embedding_heads_v) *
                                @as(f64, heads_kv) *
                                bytes_per_element);
                        }
                    }
                }
            }
        }

        // TODO: handle other architectures...

        return .{
            .kv = kv,
            .partial_offload = partial_offload,
            .full_offload = full_offload,
        };
    }

    pub fn supportsFlashAttention(self: Self) bool {
        const arch = self.KV.architecture();
        const key = std.fmt.allocPrint(std.heap.page_allocator, "{s}.pooling_type", .{arch}) catch return false;
        defer std.heap.page_allocator.free(key);

        if (self.KV.KV.get(key) != null) {
            return false;
        }

        const head_count_k = self.KV.EmbeddingHeadCountK();
        const head_count_v = self.KV.EmbeddingHeadCountV();

        return head_count_k != 0 and head_count_v != 0 and head_count_k == head_count_v;
    }
};

pub fn kvCacheBytesPerElement(cache_type: []const u8) f64 {
    if (std.mem.eql(u8, cache_type, "q8_0")) {
        return 1.0; // 1/2 of fp16
    } else if (std.mem.eql(u8, cache_type, "q4_0")) {
        return 0.5; // 1/4 of fp16
    } else {
        return 2.0; // f16 (default)
    }
}

const Container = struct {
    nameFn: fn (self: *const anyopaque) []const u8,
    decodeFn: fn (self: *anyopaque, stream: anytype) Model,
    const Self = @This();

    pub fn name(self: *Self) []const u8 {
        return self.nameFn(self);
    }

    pub fn decode(self: *Self, stream: anytype) !Model {
        return self.decodeFn(self, stream);
    }
};
const Model = struct {
    self: *anyopaque,
    kvFn: fn (self: *anyopaque) KV,
    tensorsFn: fn (self: *anyopaque) Tensors,
    const Self = @This();
    pub fn kv(self: *Self) KV {
        return self.kvFn(self.self);
    }

    pub fn tensors(self: *Self) Tensors {
        return self.tensorsFn(self.self);
    }
};
