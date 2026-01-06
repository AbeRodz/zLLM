const std = @import("std");
const types = @import("sampler/types.zig");
const common_types = @import("./common/types.zig");
const enums = @import("sampler/enums.zig");
const common_init = @import("llama_common_init.zig");
const common_params = @import("llama_common.zig");
const llama = @cImport({
    @cInclude("llama.h");
});

pub const NgramData = struct {
    actve: bool = false,
    seq_id: llama.llama_seq_id = -1,
    i_batch: std.ArrayList(i32),
    tokens: std.ArrayList(llama.llama_token),
};

pub const NgramContainer = struct {
    n_total: i32 = 0,
    count: std.ArrayList(i32),
    head: std.ArrayList(i32),
    tokens: std.ArrayList(llama.llama_token),

    pub fn init(allocator: std.mem.Allocator, n_vocab: i32, N: i32, G: i32) !NgramContainer {
        return NgramContainer{
            .count = std.ArrayList(i32).initCapacity(
                allocator,
                n_vocab,
            ),
            .head = std.ArrayList(i32).init(
                allocator,
                n_vocab,
            ),
            .tokens = std.ArrayList(llama.llama_token).initCapacity(
                allocator,
                n_vocab * G * (N - 1),
            ),
        };
    }
    pub fn deinit(self: *NgramContainer) void {
        self.count.deinit();
        self.head.deinit();
        self.tokens.deinit();
    }
};

pub fn main(){
    //common_init
    const params : common_types.CommonParams = undefined;

    const  W = 15; // lookahead window
    const  N = 5;  // n-gram size
    const  G = 15; // max verification n-grams
    const dump_kv_cache = params.dump_kv_cache;
    llama.llama_backend_init();
    llama.llama_numa_init(params.numa);

    const llama_init =common_init.common_init_from_params(params);

    const model =  llama_init.model;
    const ctx = llama_init.context;
    const vocab = llama.llama_model_get_vocab(model).?;
    
}