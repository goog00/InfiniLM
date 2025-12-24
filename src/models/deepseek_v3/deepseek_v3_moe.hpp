#ifndef DEEPSEEK_V3_MOE_H
#define DEEPSEEK_V3_MOE_H

#include "deepseek_v3_impl.hpp"
#include "../../layers/fused_linear.hpp"
#include "../../tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm {

class Gate {
public:
    Gate(const GateWeight* weight, infiniDevice_t device, int n_routed_experts, int n_activated_experts,
         float route_scale, const std::string& score_func = "softmax", int n_groups = 1, int topk_groups = 1);

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward(const std::shared_ptr<Tensor>& x);

private:
    const GateWeight* weight_;
    infiniDevice_t device_;
    int n_routed_experts_;
    int n_activated_experts_;
    float route_scale_;
    std::string score_func_;
    int n_groups_;
    int topk_groups_;
};

class Expert {
public:
    Expert(const ExpertWeight* weight, infiniDevice_t device, int dim, int inter_dim);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

private:
    const ExpertWeight* weight_;
    infiniDevice_t device_;
    int dim_;
    int inter_dim_;
    std::shared_ptr<layers::FP8Linear> gate_proj_;
    std::shared_ptr<layers::FP8Linear> up_proj_;
    std::shared_ptr<layers::FP8Linear> down_proj_;
};

class MoE {
public:
    MoE(const MoEWeight* weight, infiniDevice_t device, int dim, int inter_dim,
        int n_routed_experts, int n_activated_experts, float route_scale,
        int n_shared_experts = 1);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

private:
    const MoEWeight* weight_;
    infiniDevice_t device_;
    int dim_;
    int inter_dim_;
    int n_routed_experts_;
    int n_activated_experts_;
    float route_scale_;
    int n_shared_experts_;
    std::unique_ptr<Gate> gate_;
    std::vector<std::unique_ptr<Expert>> experts_;
    std::unique_ptr<Expert> shared_expert_;  // Assuming MLP for shared
};

} // namespace infinilm

#endif