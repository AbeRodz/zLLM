pub const llama = @cImport({
    @cInclude("llama.h");
});

pub const LlamaModel = llama.struct_llama_model;
pub const Params = llama.struct_llama_model_params;

pub fn loadModel(gguf_path: []const u8, params: llama.llama_model_params) ?*LlamaModel {
    return llama.llama_model_load_from_file(gguf_path.ptr, params);
}

pub fn initModel(model: *LlamaModel, params: llama.llama_model_params) ?*llama.struct_llama_context {
    return llama.llama_init_from_model(model, params);
}
pub fn default_params() Params {
    return llama.llama_model_default_params();
}
