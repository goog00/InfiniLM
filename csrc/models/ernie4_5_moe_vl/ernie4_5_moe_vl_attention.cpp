#include "ernie4_5_moe_vl_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5_VLMoeAttention::Ernie4_5_VLMoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 size_t layer_idx,
                                                 const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim"); // supplied by create_ernie4_5_moe_vl_model_config

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("use_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % tp_size))) {
        throw std::runtime_error("Ernie4_5_VLMoeAttention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    // TODO(ernie-vl): get_rope must build a 3D / mrope-aware RoPE from
    // config.rope_scaling.mrope_section = [22, 22, 20] and rope_theta = 500000.
    // Verify infinicore::nn::RoPE supports mrope sections; if not, a custom
    // apply_rotary_3d is required here (see modeling_ernie4_5_vl.apply_rotary_3d).
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Ernie4_5_VLMoeAttention::forward(const infinicore::Tensor &positions,
                                                    const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Ernie4_5_VLMoeAttention::forward_static_(const infinicore::Tensor &position_ids,
                                                            const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // TODO(ernie-vl): 3D mrope. position_ids is [3, batch, seq] = (time, height, width).
    // mrope_section=[22,22,20] splits head_dim=128 into three rotary segments.
    // For now fall back to 1D rope using the time dimension (pos[0]) as a placeholder.
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    }

    // Write RoPE into a fresh contiguous buffer matching the layout expected by
    // StaticAttentionImpl ([batch, heads, seq, dim] permuted to [batch, seq, heads, dim]).
    // Applying RoPE in-place on the narrowed (non-contiguous) q_reshaped leaves strides
    // incompatible with the view inside StaticAttentionImpl.
    auto q_rope = infinicore::Tensor::empty(
        {batch_size, num_attention_heads_, seq_len, head_dim_},
        q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Ernie4_5_VLMoeAttention::forward_paged_(const infinicore::Tensor &position_ids,
                                                           const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // Make contiguous before view: q/k/v are narrow slices of the fused QKV output
    // and therefore non-contiguous; view on a non-contiguous tensor can fail in paged
    // attention backends that permute+view the result.
    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();

    auto q_reshaped = q_cont->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k_cont->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v_cont->view({seq_len, num_key_value_heads_, head_dim_});

    // TODO(ernie-vl): 3D mrope (see forward_static_).
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::ernie4_5_moe_vl
