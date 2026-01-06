const std = @import("std");
const enums = @import("enums.zig");
const CommonParamsSampling = @import("../llama_sampler.zig").CommonParamsSampling;
pub const llama = @cImport({
    @cInclude("llama.h");
});
const common_grammar_trigger = struct {
    trigger_type: enums.common_grammar_trigger_type,
    value: []const u8,
    token: llama.llama_token = llama.LLAMA_TOKEN_NULL,
};

const common_params_speculative = struct {
    devices: std.ArrayList(llama.ggml_backend_dev_t) = .empty,
    n_ctx: i32 = 0, // draft context size
    n_max: i32 = 16, // max draft tokens
    n_min: i32 = 0, // min draft tokens
    n_gpu_layers: i32 = -1, // VRAM layers for draft model (-1 = default)
    p_split: f32 = 0.1, // speculative split probability
    p_min: f32 = 0.75, // min speculative decoding probability
    cpuparams: cpu_params = cpu_params{},
    cpuparams_batch: cpu_params = cpu_params{},
    model: common_params_model = common_params_model{},
};

const common_params_vocoder = struct {
    model: common_params_model = common_params_model{},
    speaker_file: []const u8 = "", // speaker file path
    use_guide_tokens: bool = false,
};

const common_params_model = struct {
    path: []const u8 = "", // model local path
    url: []const u8 = "", // model url to download
    hf_repo: []const u8 = "", // HF repo
    hf_file: []const u8 = "", // HF file
};

const common_adapter_lora_info = struct {
    path: []const u8,
    scale: f32,
    ptr: *llama.llama_adapter_lora,
};
const common_control_vector_data = struct {
    n_embd: i32,
    data: std.ArrayList(f32) = .empty,
};
const common_control_vector_load_info = struct {
    strength: f32,
    fname: []const u8,
};

const cpu_params = struct {
    n_threads: i32 = -1,
    cpumask: [llama.GGML_MAX_N_THREADS]bool = [_]bool{false} ** llama.GGML_MAX_N_THREADS,
    mask_valid: bool = false,
    priority: llama.enum_ggml_sched_priority = llama.GGML_SCHED_PRIO_NORMAL,
    strict_cpu: bool = false,
    poll: u32 = 50,
};

pub const CommonParams = struct {
    n_predict: i32 = -1,
    n_ctx: i32 = 4096,
    n_batch: i32 = 2048,
    n_ubatch: i32 = 512,
    n_keep: i32 = 0,
    n_chunks: i32 = -1,
    n_parallel: i32 = 1,
    n_sequences: i32 = 1,
    grp_attn_n: i32 = 1,
    grp_attn_w: i32 = 512,
    n_print: i32 = -1,
    rope_freq_base: f32 = 0.0,
    rope_freq_scale: f32 = 0.0,
    yarn_ext_factor: f32 = -1.0,
    yarn_attn_factor: f32 = 1.0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    yarn_orig_ctx: i32 = 0,
    defrag_thold: f32 = 0.1,
    devices: []llama.ggml_backend_dev_t = undefined,
    n_gpu_layers: i32 = -1,
    main_gpu: i32 = 0,
    tensor_split: [128]f32 = [_]f32{0} ** 128,
    split_mode: llama.enum_llama_split_mode = llama.LLAMA_SPLIT_MODE_LAYER,
    cpuparams: cpu_params = cpu_params{},
    cpuparams_batch: cpu_params = cpu_params{},
    cb_eval: ?llama.ggml_backend_sched_eval_callback = null,
    cb_eval_user_data: ?*anyopaque = null,
    numa: llama.enum_ggml_numa_strategy = llama.GGML_NUMA_STRATEGY_DISABLED,
    rope_scaling_type: llama.llama_rope_scaling_type = llama.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
    pooling_type: llama.enum_llama_pooling_type = llama.LLAMA_POOLING_TYPE_UNSPECIFIED,
    attention_type: llama.llama_attention_type = llama.LLAMA_ATTENTION_TYPE_UNSPECIFIED,
    sampling: CommonParamsSampling = CommonParamsSampling{},
    speculative: common_params_speculative = common_params_speculative{},
    vocoder: common_params_vocoder = common_params_vocoder{},
    model: common_params_model = common_params_model{},
    model_alias: []const u8 = "",
    hf_token: []const u8 = "",
    prompt: []const u8 = "",
    system_prompt: []const u8 = "",
    prompt_file: []const u8 = "",
    path_prompt_cache: []const u8 = "",
    input_prefix: []const u8 = "",
    input_suffix: []const u8 = "",
    lookup_cache_static: []const u8 = "",
    lookup_cache_dynamic: []const u8 = "",
    logits_file: []const u8 = "",
    in_files: std.ArrayList([]const u8) = .empty,
    antiprompt: std.ArrayList([]const u8) = .empty,
    kv_overrides: []llama.llama_model_kv_override = undefined,
    tensor_buft_overrides: []llama.llama_model_tensor_buft_override = undefined,
    lora_init_without_apply: bool = false,
    lora_adapters: []common_adapter_lora_info = undefined,
    control_vectors: []common_control_vector_load_info = undefined,
    verbosity: i32 = 0,
    control_vector_layer_start: i32 = -1,
    control_vector_layer_end: i32 = -1,
    ppl_stride: i32 = 0,
    ppl_output_type: i32 = 0,
    hellaswag: bool = false,
    hellaswag_tasks: usize = 400,
    winogrande: bool = false,
    winogrande_tasks: usize = 0,
    multiple_choice: bool = false,
    multiple_choice_tasks: usize = 0,
    kl_divergence: bool = false,
    usage: bool = false,
    completion: bool = false,
    use_color: bool = false,
    special: bool = false,
    interactive: bool = false,
    interactive_first: bool = false,
    prompt_cache_all: bool = false,
    prompt_cache_ro: bool = false,
    escape: bool = true,
    multiline_input: bool = false,
    simple_io: bool = false,
    cont_batching: bool = true,
    flash_attn: bool = false,
    no_perf: bool = false,
    ctx_shift: bool = true,
    input_prefix_bos: bool = false,
    logits_all: bool = false,
    use_mmap: bool = true,
    use_mlock: bool = false,
    verbose_prompt: bool = false,
    display_prompt: bool = true,
    dump_kv_cache: bool = false,
    no_kv_offload: bool = false,
    warmup: bool = true,
    check_tensors: bool = false,
    single_turn: bool = false,
    cache_type_k: llama.enum_ggml_type = llama.GGML_TYPE_F16,
    cache_type_v: llama.enum_ggml_type = llama.GGML_TYPE_F16,
    conversation_mode: enums.common_conversation_mode = .COMMON_CONVERSATION_MODE_AUTO,
    mmproj: common_params_model = common_params_model{},
    image: std.ArrayList([]const u8) = .empty,
    // embedding
    embedding: bool = false,
    embd_normalize: i32 = 2,
    embd_out: []const u8 = "",
    embd_sep: []const u8 = "\n",
    reranking: bool = false,
    // server
    port: i32 = 8080,
    timeout_read: i32 = 600,
    timeout_write: i32 = 600,
    n_threads_http: i32 = -1,
    n_cache_reuse: i32 = 0,
    hostname: []const u8 = "127.0.0.1",
    public_path: []const u8 = "",
    chat_template: []const u8 = "",
    use_jinja: bool = false,
    enable_chat_template: bool = true,
    reasoning_format: enums.common_reasoning_format = .COMMON_REASONING_FORMAT_DEEPSEEK,
    api_keys: std.ArrayList([]const u8) = .empty,
    ssl_file_key: []const u8 = "",
    ssl_file_cert: []const u8 = "",
    webui: bool = true,
    endpoint_slots: bool = false,
    endpoint_props: bool = false,
    endpoint_metrics: bool = false,
    log_json: bool = false,
    slot_save_path: []const u8 = undefined,
    slot_prompt_similarity: f32 = 0.5,
    // batched-bench params
    is_pp_shared: bool = false,
    n_pp: std.ArrayList([]i32) = .empty,
    n_tg: std.ArrayList([]i32) = .empty,
    n_pl: std.ArrayList([]i32) = .empty,
    // retrieval params
    context_files: std.ArrayList([]const u8) = .empty,
    chunk_size: i32 = 64,
    chunk_separator: []const u8 = "\n",
    // passkey params
    n_junk: i32 = 250,
    i_pos: i32 = -1,
    // imatrix params
    n_out_freq: i32 = 10,
    n_save_freq: i32 = 0,
    i_chunk: i32 = 0,
    process_output: bool = false,
    compute_ppl: bool = true,
    // cvector-generator params
    n_pca_batch: i32 = 100,
    n_pca_iterations: i32 = 1000,
    cvector_dimre_method: enums.dimre_method = .DIMRE_METHOD_PCA,
    cvector_positive_file: []const u8 = "examples/cvector-generator/positive.txt",
    cvector_negative_file: []const u8 = "examples/cvector-generator/negative.txt",
    spm_infill: bool = false,
    batched_bench_output_jsonl: bool = false,
    out_file: []const u8 = undefined,
};
