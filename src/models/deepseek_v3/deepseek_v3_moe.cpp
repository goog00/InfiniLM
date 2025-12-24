#include "deepseek_v3_moe.hpp"
#include "../../utils.hpp"
#include <algorithm>
#include <cmath>

namespace infinilm {

// ---------------------------------------------------------
// Gate
// ---------------------------------------------------------
Gate::Gate(const GateWeight* weight, infiniDevice_t device, int n_routed_experts, int n_activated_experts,
           float route_scale, const std::string& score_func, int n_groups, int topk_groups)
    : weight_(weight), device_(device), n_routed_experts_(n_routed_experts),
      n_activated_experts_(n_activated_experts), route_scale_(route_scale),
      score_func_(score_func), n_groups_(n_groups), topk_groups_(topk_groups) {}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> Gate::forward(const std::shared_ptr<Tensor>& x) {
    // Compute scores = linear(x, weight)
    // For simplicity, assume Tensor has linear operation
    auto scores = linear(x, weight_->w, weight_->b);

    // Apply score function
    if (score_func_ == "softmax") {
        scores = softmax(scores);
    } else if (score_func_ == "sigmoid") {
        scores = sigmoid(scores);
    }

    // Handle groups if n_groups > 1
    if (n_groups_ > 1) {
        // Group scores and select topk_groups
        // Simplified: assume scores shape is (batch*seq, n_routed_experts)
        // TODO: implement grouping logic
    }

    // Top-k selection
    auto [weights, indices] = topk(scores, n_activated_experts_);

    // Scale weights
    weights = mul(weights, route_scale_);

    if (score_func_ == "sigmoid") {
        // Normalize weights
        weights = div(weights, sum(weights, -1, true));
    }

    return {weights, indices};
}

// ---------------------------------------------------------
// Expert
// ---------------------------------------------------------
Expert::Expert(const ExpertWeight* weight, infiniDevice_t device, int dim, int inter_dim)
    : weight_(weight), device_(device), dim_(dim), inter_dim_(inter_dim) {
    // Initialize FP8 Linear layers
    gate_proj_ = std::make_shared<layers::FP8Linear>(dim, inter_dim, false, infinicore::DataType::F8_E4M3, device);
    up_proj_ = std::make_shared<layers::FP8Linear>(dim, inter_dim, false, infinicore::DataType::F8_E4M3, device);
    down_proj_ = std::make_shared<layers::FP8Linear>(inter_dim, dim, false, infinicore::DataType::F8_E4M3, device);

    // Load weights (simplified)
    // TODO: load from weight_->gate, etc.
}

std::shared_ptr<Tensor> Expert::forward(const std::shared_ptr<Tensor>& x) {
    // Standard MLP: down(silu(gate(x)) * up(x))
    auto gate_out = gate_proj_->forward(*x);
    gate_out = silu(gate_out);
    auto up_out = up_proj_->forward(*x);
    auto combined = mul(gate_out, up_out);
    auto out = down_proj_->forward(combined);
    return out;
}

// ---------------------------------------------------------
// MoE
// ---------------------------------------------------------
MoE::MoE(const MoEWeight* weight, infiniDevice_t device, int dim, int inter_dim,
         int n_routed_experts, int n_activated_experts, float route_scale, int n_shared_experts)
    : weight_(weight), device_(device), dim_(dim), inter_dim_(inter_dim),
      n_routed_experts_(n_routed_experts), n_activated_experts_(n_activated_experts),
      route_scale_(route_scale), n_shared_experts_(n_shared_experts) {
    gate_ = std::make_unique<Gate>(weight_->route.get(), device_, n_routed_experts_, n_activated_experts_, route_scale_);

    experts_.reserve(n_routed_experts_);
    for (int i = 0; i < n_routed_experts_; ++i) {
        experts_.emplace_back(std::make_unique<Expert>(weight_->experts[i].get(), device_, dim_, inter_dim));
    }

    // Shared expert (assuming it's an MLP, similar to Expert)
    shared_expert_ = std::make_unique<Expert>(reinterpret_cast<const ExpertWeight*>(weight_->shared_expert.get()), device_, dim_, inter_dim * n_shared_experts);
}

std::shared_ptr<Tensor> MoE::forward(const std::shared_ptr<Tensor>& x) {
    auto shape = x->shape();
    auto x_flat = reshape(x, {-1, dim_});

    auto [weights, indices] = gate_->forward(x_flat);

    // Initialize output
    auto y = zeros_like(x_flat);

    // Count activations per expert
    std::vector<int> counts(n_routed_experts_, 0);
    for (int i = 0; i < indices->size(0); ++i) {
        for (int j = 0; j < n_activated_experts_; ++j) {
            int idx = indices->data<int>()[i * n_activated_experts_ + j];
            counts[idx]++;
        }
    }

    // Forward through experts
    for (int i = 0; i < n_routed_experts_; ++i) {
        if (counts[i] == 0) continue;

        // Gather inputs for this expert
        // TODO: implement gather based on indices

        auto expert_out = experts_[i]->forward(x_flat);  // Simplified
        // Scatter back
        // TODO: scatter_add to y
    }

    // Shared expert
    auto shared_out = shared_expert_->forward(x_flat);
    y = add(y, shared_out);

    // All-reduce if distributed
    // TODO: all_reduce(y)

    auto output = reshape(y, shape);
    return output;
}

} // namespace infinilm