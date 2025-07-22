// This is a Zig translation of the Python code that defines a mapping from MODEL_TENSOR to string sequences.
// The original Python relies on enums and dictionaries with tuple values, translated here using enums and struct literals.

const std = @import("std");
const ModelTensor = @import("constants.zig").ModelTensor;
pub const TensorNameMap = struct {
    pub const Mappings = std.AutoHashMap(ModelTensor, []const []const u8);

    const Self = @This();

    mappings: Mappings,
    block_mappings: Mappings,

    pub fn init(allocator: std.mem.Allocator) !TensorNameMap {
        var map = Mappings.init(allocator);
        var block_map = Mappings.init(allocator);

        try map.put(ModelTensor.TOKEN_EMBD, &[_][]const u8{
            "gpt_neox.embed_in", // gptneox
            "transformer.wte", // gpt2 gpt-j mpt refact qwen dbrx jais exaone
            "transformer.word_embeddings", // falcon
            "word_embeddings", // bloom
            "model.embed_tokens", // llama-hf nemotron olmoe olmo2
            "tok_embeddings", // llama-pth
            "embeddings.word_embeddings", // bert nomic-bert
            "language_model.embedding.word_embeddings", // persimmon
            "wte", // gpt2
            "transformer.embd.wte", // phi2
            "model.tok_embeddings", // internlm2
            "model.embedding", // mamba-qbert
            "backbone.embedding", // mamba
            "backbone.embeddings", // mamba-hf
            "transformer.in_out_embed", // Grok
            "embedding.word_embeddings", // chatglm
            "transformer.token_embeddings", // openelm
            "shared", // t5
            "rwkv.embeddings", // rwkv
        });
        try map.put(ModelTensor.TOKEN_TYPES, &[_][]const u8{
            "embeddings.token_type_embeddings",
        });

        try map.put(ModelTensor.TOKEN_EMBD_NORM, &[_][]const u8{
            "word_embeddings_layernorm", // bloom
            "embeddings.LayerNorm", // bert
            "emb_ln", // nomic-bert
            "transformer.norm", // openelm
            "rwkv.blocks.0.pre_ln", // rwkv
            "backbone.norm", // wavtokenizer
        });
        try map.put(ModelTensor.POS_EMBD, &[_][]const u8{
            "transformer.wpe", // gpt2
            "embeddings.position_embeddings", // bert
            "wpe", // gpt2
        });
        try map.put(ModelTensor.OUTPUT, &[_][]const u8{
            "embed_out", // gptneox
            "lm_head", // gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx jais nemotron exaone olmoe olmo2
            "output", // llama-pth bloom internlm2
            "word_embeddings_for_head", // persimmon
            "lm_head.linear", // phi2
            "output_layer", // chatglm
            "head", // rwkv
            "head.out", // wavtokenizer
        });
        try map.put(ModelTensor.OUTPUT_NORM, &[_][]const u8{
            "gpt_neox.final_layer_norm", // gptneox
            "transformer.ln_f", // gpt2 gpt-j falcon jais exaone
            "model.norm", // llama-hf baichuan internlm2 olmoe olmo2
            "norm", // llama-pth
            "transformer.norm_f", // mpt dbrx
            "ln_f", // refact bloom qwen gpt2
            "language_model.encoder.final_layernorm", // persimmon
            "model.final_layernorm", // persimmon
            "lm_head.ln", // phi2
            "model.norm_f", // mamba-qbert
            "backbone.norm_f", // mamba
            "transformer.rms_norm", // Grok
            "encoder.final_layernorm", // chatglm
            "transformer.norm", // openelm
            "model.norm", // nemotron
            "rwkv.ln_out", // rwkv
            "backbone.final_layer_norm", // wavtokenizer
        });
        try map.put(ModelTensor.OUTPUT_NORM, &[_][]const u8{
            "rope.freqs", // llama-pth
            "rotary_pos_emb.inv_freq", // chatglm
        });
        try map.put(ModelTensor.CONV1D, &[_][]const u8{
            "backbone.embed", // roberta
        });

        try block_map.put(ModelTensor.ATTN_NORM, &[_][]const u8{
            "gpt_neox.layers.{}.input_layernorm", // gptneox
            "transformer.h.{}.ln_1", // gpt2 gpt-j refact qwen jais exaone
            "transformer.blocks.{}.norm_1", // mpt
            "transformer.h.{}.input_layernorm", // falcon7b
            "h.{}.input_layernorm", // bloom
            "transformer.h.{}.ln_mlp", // falcon40b
            "model.layers.{}.input_layernorm", // llama-hf nemotron olmoe
            "layers.{}.attention_norm", // llama-pth
            "language_model.encoder.layers.{}.input_layernorm", // persimmon
            "model.layers.{}.ln1", // yi
            "h.{}.ln_1", // gpt2
            "transformer.h.{}.ln", // phi2
            "model.layers.layers.{}.norm", // plamo
            "model.layers.{}.attention_norm", // internlm2
            "model.layers.{}.norm", // mamba-qbert
            "backbone.layers.{}.norm", // mamba
            "transformer.decoder_layer.{}.rms_norm", // Grok
            "transformer.blocks.{}.norm_attn_norm.norm_1", // dbrx
            "encoder.layers.{}.input_layernorm", // chatglm
            "transformer.layers.{}.attn_norm", // openelm
            "rwkv.blocks.{}.ln1", // rwkv
        });

        try block_map.put(ModelTensor.ATTN_NORM_2, &[_][]const u8{
            "transformer.h.{}.ln_attn", // falcon40b
            "encoder.layer.{}.layer_norm_1", // jina-v2-code
            "rwkv.blocks.{}.ln2", // rwkv
        });
        try block_map.put(ModelTensor.ATTN_QKV, &[_][]const u8{
            "gpt_neox.layers.{}.attention.query_key_value", // gptneox
            "transformer.h.{}.attn.c_attn", // gpt2 qwen jais
            "transformer.blocks.{}.attn.Wqkv", // mpt
            "transformer.blocks.{}.norm_attn_norm.attn.Wqkv", // dbrx
            "transformer.h.{}.self_attention.query_key_value", // falcon
            "h.{}.self_attention.query_key_value", // bloom
            "language_model.encoder.layers.{}.self_attention.query_key_value", // persimmon
            "model.layers.{}.self_attn.query_key_value", // persimmon
            "h.{}.attn.c_attn", // gpt2
            "transformer.h.{}.mixer.Wqkv", // phi2
            "encoder.layers.{}.attn.Wqkv", // nomic-bert
            "model.layers.{}.self_attn.qkv_proj", // phi3
            "encoder.layers.{}.self_attention.query_key_value", // chatglm
            "transformer.layers.{}.attn.qkv_proj",
        });
        try block_map.put(ModelTensor.ATTN_Q, &[_][]const u8{
            "model.layers.{}.self_attn.q_proj", // llama-hf nemotron olmoe olmo2
            "model.layers.{}.self_attn.q_proj_no_perm", // llama-custom
            "layers.{}.attention.wq", // llama-pth
            "encoder.layer.{}.attention.self.query", // bert
            "transformer.h.{}.attn.q_proj", // gpt-j
            "model.layers.layers.{}.self_attn.q_proj", // plamo
            "model.layers.{}.attention.wq", // internlm2
            "transformer.decoder_layer.{}.multi_head_attention.query", // Grok
            "transformer.h.{}.attn.attention.q_proj",
        });
        try block_map.put(ModelTensor.ATTN_K, &[_][]const u8{
            "model.layers.{}.self_attn.k_proj", // llama-hf nemotron olmoe olmo2
            "model.layers.{}.self_attn.k_proj_no_perm", // llama-custom
            "layers.{}.attention.wk", // llama-pth
            "encoder.layer.{}.attention.self.key", // bert
            "transformer.h.{}.attn.k_proj", // gpt-j
            "transformer.h.{}.attn.k", // refact
            "model.layers.layers.{}.self_attn.k_proj", // plamo
            "model.layers.{}.attention.wk", // internlm2
            "transformer.decoder_layer.{}.multi_head_attention.key", // Grok
            "transformer.h.{}.attn.attention.k_proj",
        });
        try block_map.put(ModelTensor.ATTN_V, &[_][]const u8{
            "model.layers.{}.self_attn.v_proj", // llama-hf nemotron olmoe olmo2
            "layers.{}.attention.wv", // llama-pth
            "encoder.layer.{}.attention.self.value", // bert
            "transformer.h.{}.attn.v_proj", // gpt-j
            "transformer.h.{}.attn.v", // refact
            "model.layers.layers.{}.self_attn.v_proj", // plamo
            "model.layers.{}.attention.wv", // internlm2
            "transformer.decoder_layer.{}.multi_head_attention.value", // Grok
            "transformer.h.{}.attn.attention.v_proj",
        });
        try block_map.put(ModelTensor.ATTN_OUT, &[_][]const u8{
            "gpt_neox.layers.{}.attention.dense", // gptneox
            "transformer.h.{}.attn.c_proj", // gpt2 refact qwen jais
            "transformer.blocks.{}.attn.out_proj", // mpt
            "transformer.h.{}.self_attention.dense", // falcon
            "h.{}.self_attention.dense", // bloom
            "model.layers.{}.self_attn.o_proj", // llama-hf nemotron olmoe olmo2
            "model.layers.{}.self_attn.linear_attn", // deci
            "layers.{}.attention.wo", // llama-pth
            "encoder.layer.{}.attention.output.dense", // bert
            "transformer.h.{}.attn.out_proj", // gpt-j
            "language_model.encoder.layers.{}.self_attention.dense", // persimmon
            "model.layers.{}.self_attn.dense", // persimmon
            "h.{}.attn.c_proj", // gpt2
            "transformer.h.{}.mixer.out_proj", // phi2
            "model.layers.layers.{}.self_attn.o_proj", // plamo
            "model.layers.{}.attention.wo", // internlm2
            "encoder.layers.{}.attn.out_proj", // nomic-bert
            "transformer.decoder_layer.{}.multi_head_attention.linear", // Grok
            "transformer.blocks.{}.norm_attn_norm.attn.out_proj", // dbrx
            "encoder.layers.{}.self_attention.dense", // chatglm
            "transformer.layers.{}.attn.out_proj", // openelm
            "transformer.h.{}.attn.attention.out_proj",
        });
        try block_map.put(ModelTensor.ATTN_OUT_NORM, &[_][]const u8{
            "encoder.layer.{}.attention.output.LayerNorm", // bert
            "encoder.layers.{}.norm1", // nomic-bert
            "transformer.decoder_layer.{}.rms_norm_1", // Grok
            "transformer.blocks.{}.norm_attn_norm.norm_2", // dbrx
        });
        try block_map.put(ModelTensor.ATTN_POST_NORM, &[_][]const u8{
            "model.layers.{}.post_attention_layernorm", // gemma2 olmo2
        });
        try block_map.put(ModelTensor.ATTN_ROT_EMBD, &[_][]const u8{
            "model.layers.{}.self_attn.rotary_emb.inv_freq", // llama-hf
            "layers.{}.attention.inner_attention.rope.freqs", // llama-pth
            "model.layers.layers.{}.self_attn.rotary_emb.inv_freq", // plamo
            "transformer.h.{}.attn.rotary_emb.inv_freq", // codeshell
        });
        try block_map.put(ModelTensor.FFN_NORM, &[_][]const u8{
            "gpt_neox.layers.{}.post_attention_layernorm", // gptneox
            "transformer.h.{}.ln_2", // gpt2 refact qwen jais exaone
            "h.{}.post_attention_layernorm", // bloom
            "transformer.blocks.{}.norm_2", // mpt
            "model.layers.{}.post_attention_layernorm", // llama-hf nemotron olmoe
            "layers.{}.ffn_norm", // llama-pth
            "language_model.encoder.layers.{}.post_attention_layernorm", // persimmon
            "model.layers.{}.ln2", // yi
            "h.{}.ln_2", // gpt2
            "model.layers.{}.ffn_norm", // internlm2
            "transformer.decoder_layer.{}.rms_norm_2", // Grok
            "encoder.layers.{}.post_attention_layernorm", // chatglm
            "transformer.layers.{}.ffn_norm", // openelm
        });
        try block_map.put(ModelTensor.FFN_PRE_NORM, &[_][]const u8{
            "model.layers.{}.pre_feedforward_layernorm", // gemma2
        });
        try block_map.put(ModelTensor.FFN_POST_NORM, &[_][]const u8{
            "model.layers.{}.post_feedforward_layernorm", // gemma2 olmo2
        });
        try block_map.put(ModelTensor.FFN_GATE_INP, &[_][]const u8{
            "layers.{}.feed_forward.gate", // mixtral
            "model.layers.{}.block_sparse_moe.gate", // mixtral
            "model.layers.{}.mlp.gate", // qwen2moe olmoe
            "transformer.decoder_layer.{}.router", // Grok
            "transformer.blocks.{}.ffn.router.layer", // dbrx
            "model.layers.{}.block_sparse_moe.router.layer", // granitemoe
        });
        try block_map.put(ModelTensor.FFN_GATE_INP_SHEXP, &[_][]const u8{
            "model.layers.{}.mlp.shared_expert_gate", // qwen2moe
        });
        try block_map.put(ModelTensor.FFN_EXP_PROBS_B, &[_][]const u8{
            "model.layers.{}.mlp.gate.e_score_correction", // deepseek-v3
        });

        try block_map.put(ModelTensor.FFN_UP, &[_][]const u8{
            "gpt_neox.layers.{}.mlp.dense_h_to_4h", // gptneox
            "transformer.h.{}.mlp.c_fc", // gpt2 jais
            "transformer.blocks.{}.ffn.up_proj", // mpt
            "transformer.h.{}.mlp.dense_h_to_4h", // falcon
            "h.{}.mlp.dense_h_to_4h", // bloom
            "model.layers.{}.mlp.up_proj", // llama-hf refact nemotron olmo2
            "layers.{}.feed_forward.w3", // llama-pth
            "encoder.layer.{}.intermediate.dense", // bert
            "transformer.h.{}.mlp.fc_in", // gpt-j
            "transformer.h.{}.mlp.linear_3", // refact
            "language_model.encoder.layers.{}.mlp.dense_h_to_4h", // persimmon
            "model.layers.{}.mlp.dense_h_to_4h", // persimmon
            "transformer.h.{}.mlp.w1", // qwen
            "h.{}.mlp.c_fc", // gpt2
            "transformer.h.{}.mlp.fc1", // phi2
            "model.layers.{}.mlp.fc1", // phi2
            "model.layers.{}.mlp.gate_up_proj", // phi3
            "model.layers.layers.{}.mlp.up_proj", // plamo
            "model.layers.{}.feed_forward.w3", // internlm2
            "encoder.layers.{}.mlp.fc11", // nomic-bert
            "model.layers.{}.mlp.c_fc", // starcoder2
            "encoder.layer.{}.mlp.gated_layers_v", // jina-bert-v2
            "model.layers.{}.residual_mlp.w3", // arctic
            "encoder.layers.{}.mlp.dense_h_to_4h", // chatglm
            "transformer.h.{}.mlp.c_fc_1", // exaone
        });

        try block_map.put(ModelTensor.FFN_UP_EXP, &[_][]const u8{
            "layers.{}.feed_forward.experts.w3", // mixtral (merged)
            "transformer.decoder_layer.{}.moe.linear_v", // Grok (merged)
            "transformer.blocks.{}.ffn.experts.mlp.v1", // dbrx
            "model.layers.{}.mlp.experts.up_proj", // qwen2moe olmoe (merged)
        });

        try block_map.put(ModelTensor.FFN_UP_SHEXP, &[_][]const u8{
            "model.layers.{}.mlp.shared_expert.up_proj", // qwen2moe
            "model.layers.{}.mlp.shared_experts.up_proj", // deepseek deepseek2
        });

        // AWQ-activation gate
        try block_map.put(ModelTensor.FFN_ACT, &[_][]const u8{
            "transformer.blocks.{}.ffn.act", // mpt
        });

        // Feed-forward gate
        try block_map.put(ModelTensor.FFN_GATE, &[_][]const u8{
            "model.layers.{}.mlp.gate_proj", // llama-hf refact olmo2
            "layers.{}.feed_forward.w1", // llama-pth
            "transformer.h.{}.mlp.w2", // qwen
            "transformer.h.{}.mlp.c_fc2", // jais
            "model.layers.layers.{}.mlp.gate_proj", // plamo
            "model.layers.{}.feed_forward.w1", // internlm2
            "encoder.layers.{}.mlp.fc12", // nomic-bert
            "encoder.layer.{}.mlp.gated_layers_w", // jina-bert-v2
            "transformer.h.{}.mlp.linear_1", // refact
            "model.layers.{}.residual_mlp.w1", // arctic
            "transformer.h.{}.mlp.c_fc_0", // exaone
        });

        try block_map.put(ModelTensor.FFN_GATE_EXP, &[_][]const u8{
            "layers.{}.feed_forward.experts.w1", // mixtral (merged)
            "transformer.decoder_layer.{}.moe.linear", // Grok (merged)
            "transformer.blocks.{}.ffn.experts.mlp.w1", // dbrx
            "model.layers.{}.mlp.experts.gate_proj", // qwen2moe olmoe (merged)
        });

        try block_map.put(ModelTensor.FFN_GATE_SHEXP, &[_][]const u8{
            "model.layers.{}.mlp.shared_expert.gate_proj", // qwen2moe
            "model.layers.{}.mlp.shared_experts.gate_proj", // deepseek deepseek2
        });

        // Feed-forward down
        try block_map.put(ModelTensor.FFN_DOWN, &[_][]const u8{
            "gpt_neox.layers.{}.mlp.dense_4h_to_h", // gptneox
            "transformer.h.{}.mlp.c_proj", // gpt2 refact qwen jais
            "transformer.blocks.{}.ffn.down_proj", // mpt
            "transformer.h.{}.mlp.dense_4h_to_h", // falcon
            "h.{}.mlp.dense_4h_to_h", // bloom
            "model.layers.{}.mlp.down_proj", // llama-hf nemotron olmo2
            "layers.{}.feed_forward.w2", // llama-pth
            "encoder.layer.{}.output.dense", // bert
            "transformer.h.{}.mlp.fc_out", // gpt-j
            "language_model.encoder.layers.{}.mlp.dense_4h_to_h", // persimmon
            "model.layers.{}.mlp.dense_4h_to_h", // persimmon
            "h.{}.mlp.c_proj", // gpt2
            "transformer.h.{}.mlp.fc2", // phi2
            "model.layers.{}.mlp.fc2", // phi2
            "model.layers.layers.{}.mlp.down_proj", // plamo
            "model.layers.{}.feed_forward.w2", // internlm2
            "encoder.layers.{}.mlp.fc2", // nomic-bert
            "model.layers.{}.mlp.c_proj", // starcoder2
            "encoder.layer.{}.mlp.wo", // jina-bert-v2
            "transformer.layers.{}.ffn.proj_2", // openelm
            "model.layers.{}.residual_mlp.w2", // arctic
            "encoder.layer.{}.mlp.down_layer", // jina-bert-v2
            "encoder.layers.{}.mlp.dense_4h_to_h", // chatglm
            "model.layers.h.{}.mlp.c_proj", // exaone
        });

        try block_map.put(ModelTensor.FFN_DOWN_EXP, &[_][]const u8{
            "layers.{}.feed_forward.experts.w2", // mixtral (merged)
            "transformer.decoder_layer.{}.moe.linear_1", // Grok (merged)
            "transformer.blocks.{}.ffn.experts.mlp.w2", // dbrx
            "model.layers.{}.mlp.experts.down_proj", // qwen2moe olmoe (merged)
            "model.layers.{}.block_sparse_moe.output_linear", // granitemoe
        });

        try block_map.put(ModelTensor.FFN_DOWN_SHEXP, &[_][]const u8{
            "model.layers.{}.mlp.shared_expert.down_proj", // qwen2moe
            "model.layers.{}.mlp.shared_experts.down_proj", // deepseek deepseek2
        });

        try block_map.put(ModelTensor.ATTN_Q_NORM, &[_][]const u8{
            "language_model.encoder.layers.{}.self_attention.q_layernorm",
            "model.layers.{}.self_attn.q_layernorm", // persimmon
            "model.layers.{}.self_attn.q_norm", // cohere olmoe chameleon olmo2
            "transformer.blocks.{}.attn.q_ln", // sea-lion
            "encoder.layer.{}.attention.self.layer_norm_q", // jina-bert-v2
            "transformer.layers.{}.attn.q_norm", // openelm
        });

        try block_map.put(ModelTensor.ATTN_K_NORM, &[_][]const u8{
            "language_model.encoder.layers.{}.self_attention.k_layernorm",
            "model.layers.{}.self_attn.k_layernorm", // persimmon
            "model.layers.{}.self_attn.k_norm", // cohere olmoe chameleon olmo2
            "transformer.blocks.{}.attn.k_ln", // sea-lion
            "encoder.layer.{}.attention.self.layer_norm_k", // jina-bert-v2
            "transformer.layers.{}.attn.k_norm", // openelm
        });

        try block_map.put(ModelTensor.ROPE_FREQS, &[_][]const u8{
            "language_model.encoder.layers.{}.self_attention.rotary_emb.inv_freq", // persimmon
        });

        try block_map.put(ModelTensor.LAYER_OUT_NORM, &[_][]const u8{
            "encoder.layer.{}.output.LayerNorm", // bert
            "encoder.layers.{}.norm2", // nomic-bert
            "transformer.decoder_layer.{}.rms_norm_3", // Grok
            "encoder.layer.{}.mlp.layernorm", // jina-bert-v2
            "encoder.layer.{}.layer_norm_2", // jina-v2-code
        });

        try block_map.put(ModelTensor.SSM_IN, &[_][]const u8{
            "model.layers.{}.in_proj",
            "backbone.layers.{}.mixer.in_proj",
        });

        try block_map.put(ModelTensor.SSM_CONV1D, &[_][]const u8{
            "model.layers.{}.conv1d",
            "backbone.layers.{}.mixer.conv1d",
        });

        try block_map.put(ModelTensor.SSM_X, &[_][]const u8{
            "model.layers.{}.x_proj",
            "backbone.layers.{}.mixer.x_proj",
        });

        try block_map.put(ModelTensor.SSM_DT, &[_][]const u8{
            "model.layers.{}.dt_proj",
            "backbone.layers.{}.mixer.dt_proj",
        });

        try block_map.put(ModelTensor.SSM_A, &[_][]const u8{
            "model.layers.{}.A_log",
            "backbone.layers.{}.mixer.A_log",
        });

        try block_map.put(ModelTensor.SSM_D, &[_][]const u8{
            "model.layers.{}.D",
            "backbone.layers.{}.mixer.D",
        });

        try block_map.put(ModelTensor.SSM_OUT, &[_][]const u8{
            "model.layers.{}.out_proj",
            "backbone.layers.{}.mixer.out_proj",
        });

        try block_map.put(ModelTensor.TIME_MIX_W1, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_w1", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_W2, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_w2", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_LERP_X, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_x", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_LERP_K, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_k", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_LERP_V, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_v", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_LERP_R, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_r", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_LERP_G, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_g", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_LERP_W, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_maa_w", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_FIRST, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_faaaa", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_DECAY, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_decay", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_DECAY_W1, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_decay_w1", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_DECAY_W2, &[_][]const u8{
            "rwkv.blocks.{}.attention.time_decay_w2", // rwkv v6
        });

        try block_map.put(ModelTensor.TIME_MIX_KEY, &[_][]const u8{
            "rwkv.blocks.{}.attention.key", // rwkv
        });

        try block_map.put(ModelTensor.TIME_MIX_VALUE, &[_][]const u8{
            "rwkv.blocks.{}.attention.value", // rwkv
        });

        try block_map.put(ModelTensor.TIME_MIX_RECEPTANCE, &[_][]const u8{
            "rwkv.blocks.{}.attention.receptance", // rwkv
        });

        try block_map.put(ModelTensor.TIME_MIX_GATE, &[_][]const u8{
            "rwkv.blocks.{}.attention.gate", // rwkv
        });

        try block_map.put(ModelTensor.TIME_MIX_LN, &[_][]const u8{
            "rwkv.blocks.{}.attention.ln_x", // rwkv
        });

        try block_map.put(ModelTensor.TIME_MIX_OUTPUT, &[_][]const u8{
            "rwkv.blocks.{}.attention.output", // rwkv
        });

        try block_map.put(ModelTensor.CHANNEL_MIX_LERP_K, &[_][]const u8{
            "rwkv.blocks.{}.feed_forward.time_maa_k", // rwkv v6
        });

        try block_map.put(ModelTensor.CHANNEL_MIX_LERP_R, &[_][]const u8{
            "rwkv.blocks.{}.feed_forward.time_maa_r", // rwkv v6
        });

        try block_map.put(ModelTensor.CHANNEL_MIX_KEY, &[_][]const u8{
            "rwkv.blocks.{}.feed_forward.key", // rwkv
        });

        try block_map.put(ModelTensor.CHANNEL_MIX_RECEPTANCE, &[_][]const u8{
            "rwkv.blocks.{}.feed_forward.receptance", // rwkv
        });

        try block_map.put(ModelTensor.CHANNEL_MIX_VALUE, &[_][]const u8{
            "rwkv.blocks.{}.feed_forward.value", // rwkv
        });

        try block_map.put(ModelTensor.ATTN_Q_A, &[_][]const u8{
            "model.layers.{}.self_attn.q_a_proj", // deepseek2
        });

        try block_map.put(ModelTensor.ATTN_Q_B, &[_][]const u8{
            "model.layers.{}.self_attn.q_b_proj", // deepseek2
        });

        try block_map.put(ModelTensor.ATTN_KV_A_MQA, &[_][]const u8{
            "model.layers.{}.self_attn.kv_a_proj_with_mqa", // deepseek2
        });

        try block_map.put(ModelTensor.ATTN_KV_B, &[_][]const u8{
            "model.layers.{}.self_attn.kv_b_proj", // deepseek2
        });

        try block_map.put(ModelTensor.ATTN_Q_A_NORM, &[_][]const u8{
            "model.layers.{}.self_attn.q_a_layernorm", // deepseek2
        });

        try block_map.put(ModelTensor.ATTN_KV_A_NORM, &[_][]const u8{
            "model.layers.{}.self_attn.kv_a_layernorm", // deepseek2
        });

        try block_map.put(ModelTensor.ATTN_SUB_NORM, &[_][]const u8{
            "model.layers.{}.self_attn.inner_attn_ln", // bitnet
        });

        try block_map.put(ModelTensor.FFN_SUB_NORM, &[_][]const u8{
            "model.layers.{}.mlp.ffn_layernorm", // bitnet
        });

        try block_map.put(ModelTensor.DEC_ATTN_NORM, &[_][]const u8{
            "decoder.block.{}.layer.0.layer_norm", // t5
        });

        try block_map.put(ModelTensor.DEC_ATTN_Q, &[_][]const u8{
            "decoder.block.{}.layer.0.SelfAttention.q", // t5
        });

        try block_map.put(ModelTensor.DEC_ATTN_K, &[_][]const u8{
            "decoder.block.{}.layer.0.SelfAttention.k", // t5
        });

        try block_map.put(ModelTensor.DEC_ATTN_V, &[_][]const u8{
            "decoder.block.{}.layer.0.SelfAttention.v", // t5
        });

        try block_map.put(ModelTensor.DEC_ATTN_OUT, &[_][]const u8{
            "decoder.block.{}.layer.0.SelfAttention.o", // t5
        });

        try block_map.put(ModelTensor.DEC_ATTN_REL_B, &[_][]const u8{
            "decoder.block.{}.layer.0.SelfAttention.relative_attention_bias", // t5
        });

        try block_map.put(ModelTensor.DEC_CROSS_ATTN_NORM, &[_][]const u8{
            "decoder.block.{}.layer.1.layer_norm", // t5
        });

        try block_map.put(ModelTensor.DEC_CROSS_ATTN_Q, &[_][]const u8{
            "decoder.block.{}.layer.1.EncDecAttention.q", // t5
        });

        try block_map.put(ModelTensor.DEC_CROSS_ATTN_K, &[_][]const u8{
            "decoder.block.{}.layer.1.EncDecAttention.k", // t5
        });

        try block_map.put(ModelTensor.DEC_CROSS_ATTN_V, &[_][]const u8{
            "decoder.block.{}.layer.1.EncDecAttention.v", // t5
        });

        try block_map.put(ModelTensor.DEC_CROSS_ATTN_OUT, &[_][]const u8{
            "decoder.block.{}.layer.1.EncDecAttention.o", // t5
        });

        try block_map.put(ModelTensor.DEC_CROSS_ATTN_REL_B, &[_][]const u8{
            "decoder.block.{}.layer.1.EncDecAttention.relative_attention_bias", // t5
        });

        try block_map.put(ModelTensor.DEC_FFN_NORM, &[_][]const u8{
            "decoder.block.{}.layer.2.layer_norm", // t5
        });

        try block_map.put(ModelTensor.DEC_FFN_GATE, &[_][]const u8{
            "decoder.block.{}.layer.2.DenseReluDense.wi_0", // flan-t5
        });

        try block_map.put(ModelTensor.DEC_FFN_UP, &[_][]const u8{
            "decoder.block.{}.layer.2.DenseReluDense.wi", // t5
            "decoder.block.{}.layer.2.DenseReluDense.wi_1", // flan-t5
        });

        try block_map.put(ModelTensor.DEC_FFN_DOWN, &[_][]const u8{
            "decoder.block.{}.layer.2.DenseReluDense.wo", // t5
        });

        try block_map.put(ModelTensor.DEC_OUTPUT_NORM, &[_][]const u8{
            "decoder.final_layer_norm", // t5
        });

        try block_map.put(ModelTensor.ENC_ATTN_NORM, &[_][]const u8{
            "encoder.block.{}.layer.0.layer_norm", // t5
        });

        try block_map.put(ModelTensor.ENC_ATTN_Q, &[_][]const u8{
            "encoder.block.{}.layer.0.SelfAttention.q", // t5
        });

        try block_map.put(ModelTensor.ENC_ATTN_K, &[_][]const u8{
            "encoder.block.{}.layer.0.SelfAttention.k", // t5
        });

        try block_map.put(ModelTensor.ENC_ATTN_V, &[_][]const u8{
            "encoder.block.{}.layer.0.SelfAttention.v", // t5
        });

        try block_map.put(ModelTensor.ENC_ATTN_OUT, &[_][]const u8{
            "encoder.block.{}.layer.0.SelfAttention.o", // t5
        });

        try block_map.put(ModelTensor.ENC_ATTN_REL_B, &[_][]const u8{
            "encoder.block.{}.layer.0.SelfAttention.relative_attention_bias", // t5
        });

        try block_map.put(ModelTensor.ENC_FFN_NORM, &[_][]const u8{
            "encoder.block.{}.layer.1.layer_norm", // t5
        });

        try block_map.put(ModelTensor.ENC_FFN_GATE, &[_][]const u8{
            "encoder.block.{}.layer.1.DenseReluDense.wi_0", // flan-t5
        });

        try block_map.put(ModelTensor.ENC_FFN_UP, &[_][]const u8{
            "encoder.block.{}.layer.1.DenseReluDense.wi", // t5
            "encoder.block.{}.layer.1.DenseReluDense.wi_1", // flan-t5
        });

        try block_map.put(ModelTensor.ENC_FFN_DOWN, &[_][]const u8{
            "encoder.block.{}.layer.1.DenseReluDense.wo", // t5
        });

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // TODO: these do not belong to block_mappings_cfg - move them to mappings_cfg
        try block_map.put(ModelTensor.ENC_OUTPUT_NORM, &[_][]const u8{
            "encoder.final_layer_norm", // t5
        });

        try block_map.put(ModelTensor.CLS, &[_][]const u8{
            "classifier", // jina
            "classifier.dense", // roberta
        });

        try block_map.put(ModelTensor.CLS_OUT, &[_][]const u8{
            "classifier.out_proj", // roberta
        });
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        try block_map.put(ModelTensor.CONVNEXT_DW, &[_][]const u8{
            "backbone.convnext.{}.dwconv", // wavtokenizer
        });

        try block_map.put(ModelTensor.CONVNEXT_NORM, &[_][]const u8{
            "backbone.convnext.{}.norm", // wavtokenizer
        });

        try block_map.put(ModelTensor.CONVNEXT_PW1, &[_][]const u8{
            "backbone.convnext.{}.pwconv1", // wavtokenizer
        });

        try block_map.put(ModelTensor.CONVNEXT_PW2, &[_][]const u8{
            "backbone.convnext.{}.pwconv2", // wavtokenizer
        });

        try block_map.put(ModelTensor.CONVNEXT_GAMMA, &[_][]const u8{
            "backbone.convnext.{}.gamma", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_CONV1, &[_][]const u8{
            "backbone.posnet.{}.conv1", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_CONV2, &[_][]const u8{
            "backbone.posnet.{}.conv2", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_NORM, &[_][]const u8{
            "backbone.posnet.{}.norm", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_NORM1, &[_][]const u8{
            "backbone.posnet.{}.norm1", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_NORM2, &[_][]const u8{
            "backbone.posnet.{}.norm2", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_ATTN_NORM, &[_][]const u8{
            "backbone.posnet.{}.norm", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_ATTN_Q, &[_][]const u8{
            "backbone.posnet.{}.q", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_ATTN_K, &[_][]const u8{
            "backbone.posnet.{}.k", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_ATTN_V, &[_][]const u8{
            "backbone.posnet.{}.v", // wavtokenizer
        });

        try block_map.put(ModelTensor.POSNET_ATTN_OUT, &[_][]const u8{
            "backbone.posnet.{}.proj_out", // wavtokenizer
        });
        return map;
    }

    /// For mappings that are already concrete (no formatting needed)
    pub fn getMappings(self: *Self, tensor: ModelTensor) ?[]const []const u8 {
        return self.mappings.get(tensor);
    }

    /// For mappings with template placeholders like `{}`
    pub fn getFormattedBlockMappings(self: *Self, allocator: std.mem.Allocator, tensor: ModelTensor, bid: usize) ![]const []const u8 {
        const templates = self.block_mappings.get(tensor) orelse return &[_][]const u8{};
        var result = try allocator.alloc([]const u8, templates.len);
        for (templates, 0..) |tpl, i| {
            result[i] = try std.fmt.allocPrint(allocator, tpl, .{bid});
        }
        return result;
    }
};
