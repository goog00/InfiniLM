#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

// Text-side self-attention for ERNIE-4.5-VL-MoE.
//
// Differences vs. the generic `infinilm::layers::attention::Attention`:
//   - No q_norm / k_norm (ERNIE-4.5 attention has no QK-RMSNorm).
//   - 3D RoPE (mrope): position_ids carry [time, height, width]; rotation uses
//     `rope_scaling.mrope_section = [22, 22, 20]`.
//   - use_bias = false for all projections (config: "use_bias": false).
//
// GQA: num_attention_heads = 20, num_key_value_heads = 4, head_dim = 128.
class Ernie4_5_VLMoeAttention : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;

    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;

    // For off-line kv cache quantization.
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::ernie4_5_moe_vl
