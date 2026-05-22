#include "ernie4_5_moe_vl_moe.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

#include <stdexcept>

namespace infinilm::models::ernie4_5_moe_vl {

// ---------------------------------------------------------------------------
// Ernie4_5_VLMoeMLP
// ---------------------------------------------------------------------------
Ernie4_5_VLMoeMLP::Ernie4_5_VLMoeMLP(size_t hidden_size,
                                     size_t intermediate_size,
                                     bool use_bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size), use_bias_(use_bias) {

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    gate_proj_ = this->register_module<layers::linear::ColumnParallelLinear>(
        "gate_proj", hidden_size_, intermediate_size_, use_bias_, dtype, device, tp_rank, tp_size);
    up_proj_ = this->register_module<layers::linear::ColumnParallelLinear>(
        "up_proj", hidden_size_, intermediate_size_, use_bias_, dtype, device, tp_rank, tp_size);
    down_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "down_proj", intermediate_size_, hidden_size_, use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor Ernie4_5_VLMoeMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = hidden_states;
    auto gate = gate_proj_->forward(x);
    auto up = up_proj_->forward(x);
    auto intermediate = infinicore::op::swiglu(up, gate);
    return down_proj_->forward(intermediate);
}

// ---------------------------------------------------------------------------
// Ernie4_5_VLMoeStatics (aux-free correction bias)
// ---------------------------------------------------------------------------
Ernie4_5_VLMoeStatics::Ernie4_5_VLMoeStatics(size_t num_modalities,
                                             size_t num_experts,
                                             const infinicore::Device &device)
    : num_modalities_(num_modalities), num_experts_(num_experts) {
    // Bias is stored in fp32 in the checkpoint (load as F32).
    INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({num_modalities_, num_experts_}, infinicore::DataType::F32, device));
}

infinicore::Tensor Ernie4_5_VLMoeStatics::bias_for_modality(size_t modality) const {
    if (modality >= num_modalities_) {
        throw std::runtime_error("Ernie4_5_VLMoeStatics: modality out of range");
    }
    return e_score_correction_bias_->narrow({{0, modality, 1}})->squeeze(0);
}

// ---------------------------------------------------------------------------
// Ernie4_5_VLMoeGate
// ---------------------------------------------------------------------------
Ernie4_5_VLMoeGate::Ernie4_5_VLMoeGate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");

    // moe_num_experts = [text, vision]; moe_k = 6.
    const auto &num_experts = model_config->get_ref("moe_num_experts");
    size_t num_experts_text = num_experts[0].get<size_t>();
    size_t num_experts_vision = num_experts[1].get<size_t>();
    moe_k_ = model_config->get<size_t>("moe_k");

    // Gate weights are kept in fp32 in the checkpoint for routing precision.
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_text, hidden_size}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(weight_1, ({num_experts_vision, hidden_size}, dtype, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
Ernie4_5_VLMoeGate::forward(const infinicore::Tensor &hidden_states,
                            size_t modality,
                            const std::optional<infinicore::Tensor> &correction_bias) const {
    // hidden_states: [num_tokens, hidden_size]
    ASSERT(hidden_states->ndim() == 2);
    size_t ntoken = hidden_states->shape()[0];

    // TODO(ernie-vl): full aux-free routing should add correction_bias for
    // top-k *selection* only, then take routing weights from the un-biased
    // sigmoid scores. We currently use softmax(logits) topk (same as qwen3_moe)
    // and ignore the bias — accuracy impact is small for inference, and this is
    // the simplest path until a topk-with-bias op exists.
    (void)correction_bias;

    auto gate_weight = (modality == 0) ? weight_ : weight_1_;
    auto router_logits = infinicore::op::linear(
        const_cast<infinicore::Tensor &>(hidden_states), gate_weight, std::nullopt, 1.0f);

    auto router_scores = infinicore::Tensor::empty(
        {ntoken, moe_k_}, infinicore::DataType::F32, hidden_states->device());
    auto router_indices = infinicore::Tensor::empty(
        {ntoken, moe_k_}, infinicore::DataType::I32, hidden_states->device());

    int norm_flag = norm_topk_prob_ ? 1 : 0;
    infinicore::op::topksoftmax(router_scores, router_indices, router_logits, moe_k_, norm_flag);

    return std::make_tuple(router_scores, router_indices);
}

// ---------------------------------------------------------------------------
// Ernie4_5_VLMoeExpertList
// ---------------------------------------------------------------------------
Ernie4_5_VLMoeExpertList::Ernie4_5_VLMoeExpertList(size_t num_experts_text,
                                                    size_t num_experts_vision,
                                                    size_t hidden_size,
                                                    size_t inter_text,
                                                    size_t inter_vision,
                                                    bool use_bias,
                                                    const infinicore::DataType &dtype,
                                                    const infinicore::Device &device) {
    // Text experts: indices [0, num_experts_text) -> weight paths experts.0.*, experts.1.*, ...
    for (size_t i = 0; i < num_experts_text; ++i) {
        experts.push_back(this->register_module<Ernie4_5_VLMoeMLP>(
            std::to_string(i), hidden_size, inter_text, use_bias, dtype, device));
    }
    // Vision experts: indices [num_experts_text, total) -> weight paths experts.64.*, ...
    for (size_t i = 0; i < num_experts_vision; ++i) {
        experts.push_back(this->register_module<Ernie4_5_VLMoeMLP>(
            std::to_string(num_experts_text + i), hidden_size, inter_vision, use_bias, dtype, device));
    }
}

// ---------------------------------------------------------------------------
// Ernie4_5_VLMoeSparseMoeBlock
// ---------------------------------------------------------------------------
Ernie4_5_VLMoeSparseMoeBlock::Ernie4_5_VLMoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                           const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    bool use_bias = model_config->get_or<bool>("use_bias", false);

    const auto &num_experts = model_config->get_ref("moe_num_experts");          // [64, 64]
    const auto &intermediate = model_config->get_ref("moe_intermediate_size");    // [1536, 512]
    num_experts_text_ = num_experts[0].get<size_t>();
    num_experts_vision_ = num_experts[1].get<size_t>();
    size_t inter_text = intermediate[0].get<size_t>();
    size_t inter_vision = intermediate[1].get<size_t>();
    moe_k_ = model_config->get<size_t>("moe_k");
    size_t num_shared = model_config->get_or<size_t>("moe_num_shared_experts", 2);

    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(moe_statics, /*num_modalities=*/2, num_experts_text_, device);

    // Registered as "experts" so weight paths are mlp.experts.N.*
    // Text experts [0..num_text), vision experts [num_text..total).
    INFINICORE_NN_MODULE_INIT(experts, num_experts_text_, num_experts_vision_,
                              hidden_size, inter_text, inter_vision, use_bias, dtype, device);

    // Shared experts: one fused MLP with intermediate = num_shared * inter_text.
    // TODO(ernie-vl): confirm shared-expert layout/weight names against checkpoint.
    INFINICORE_NN_MODULE_INIT(shared_experts, hidden_size, num_shared * inter_text, use_bias, dtype, device);
}

infinicore::Tensor Ernie4_5_VLMoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states,
                                                         const infinicore::Tensor &token_type_ids) const {
    ASSERT(hidden_states->ndim() == 3);
    auto shape = hidden_states->shape();  // [batch, seq, hidden]
    size_t hidden = shape[2];
    size_t ntoken = shape[0] * shape[1];
    auto flat = hidden_states->view({ntoken, hidden});

    // 1) Shared experts: every token passes through (residual contribution).
    auto shared_out = shared_experts_->forward(flat);

    // Output starts as shared_out; routed expert outputs are added on top.
    auto final_states = infinicore::Tensor::empty(flat->shape(), flat->dtype(), flat->device());
    final_states->copy_from(shared_out);

    // 2) Read per-token modality on CPU and group token indices.
    auto tt_cpu = token_type_ids->to(infinicore::Device::cpu())->contiguous();
    const auto *tt_data = reinterpret_cast<const int64_t *>(tt_cpu->data());

    std::vector<size_t> text_idxs;
    std::vector<size_t> vision_idxs;
    text_idxs.reserve(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        if (tt_data[i] == 0) {
            text_idxs.push_back(i);
        } else {
            vision_idxs.push_back(i);
        }
    }

    // 3) Per-modality batch gate + per-token expert dispatch.
    auto dispatch_modality = [&](const std::vector<size_t> &idxs,
                                  size_t modality,
                                  size_t expert_offset) {
        if (idxs.empty()) {
            return;
        }
        size_t n = idxs.size();

        // Gather hidden states for this modality. CPU loop with narrow+copy_from is
        // not the fastest path, but keeps the implementation simple and correct.
        auto gathered = infinicore::Tensor::empty({n, hidden}, flat->dtype(), flat->device());
        for (size_t i = 0; i < n; ++i) {
            gathered->narrow({{0, i, 1}})->copy_from(flat->narrow({{0, idxs[i], 1}}));
        }

        // Batch gate forward for this modality.
        auto [weights, indices] = gate_->forward(gathered, modality, std::nullopt);
        auto w_cpu = weights->to(infinicore::Device::cpu())->contiguous();
        auto i_cpu = indices->to(infinicore::Device::cpu())->contiguous();
        const auto *w_ptr = reinterpret_cast<const float *>(w_cpu->data());
        const auto *i_ptr = reinterpret_cast<const int32_t *>(i_cpu->data());

        for (size_t i = 0; i < n; ++i) {
            auto token = gathered->narrow({{0, i, 1}});  // [1, hidden]
            infinicore::Tensor expert_sum;
            for (size_t k = 0; k < moe_k_; ++k) {
                size_t local_idx = static_cast<size_t>(i_ptr[i * moe_k_ + k]);
                size_t global_idx = expert_offset + local_idx;
                ASSERT(global_idx < experts_->experts.size());
                experts_->experts[global_idx]->set_alpha(w_ptr[i * moe_k_ + k]);
                auto out = experts_->experts[global_idx]->forward(token);
                if (k == 0) {
                    expert_sum = out;
                } else {
                    infinicore::op::add_(expert_sum, expert_sum, out);
                }
            }
            auto target = final_states->narrow({{0, idxs[i], 1}});
            infinicore::op::add_(target, target, expert_sum);
        }
    };

    dispatch_modality(text_idxs, 0, 0);                          // text experts [0 .. num_experts_text_)
    dispatch_modality(vision_idxs, 1, num_experts_text_);        // vision experts [num_experts_text_ .. total)

    return final_states->view(shape);
}

} // namespace infinilm::models::ernie4_5_moe_vl
