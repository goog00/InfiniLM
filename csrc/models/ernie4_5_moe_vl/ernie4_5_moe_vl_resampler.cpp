#include "ernie4_5_moe_vl_resampler.hpp"

#include "../../utils.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5_VLResampler::Ernie4_5_VLResampler(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};

    pixel_hidden_size_ = model_config->get<size_t>("pixel_hidden_size");      // 1280
    text_hidden_size_ = model_config->get<size_t>("hidden_size");             // 2560
    spatial_conv_size_ = model_config->get_or<size_t>("spatial_conv_size", 2);
    temporal_conv_size_ = model_config->get_or<size_t>("temporal_conv_size", 2);

    size_t spatial_dim = pixel_hidden_size_ * spatial_conv_size_ * spatial_conv_size_;  // 5120
    size_t temporal_dim = spatial_dim * temporal_conv_size_;                            // 10240
    double layer_norm_eps = 1e-6;

    // spatial_linear: Linear(5120,5120) -> act -> Linear(5120,5120) -> LayerNorm(5120)
    spatial_linear_0_ = this->register_module<infinicore::nn::Linear>(
        "spatial_linear.0", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_2_ = this->register_module<infinicore::nn::Linear>(
        "spatial_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_3_ = this->register_module<infinicore::nn::LayerNorm>(
        "spatial_linear.3", spatial_dim, layer_norm_eps, dtype, device);

    // temporal_linear: Linear(10240,5120) -> act -> Linear(5120,5120) -> LayerNorm(5120)
    temporal_linear_0_ = this->register_module<infinicore::nn::Linear>(
        "temporal_linear.0", temporal_dim, spatial_dim, true, dtype, device);
    temporal_linear_2_ = this->register_module<infinicore::nn::Linear>(
        "temporal_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    temporal_linear_3_ = this->register_module<infinicore::nn::LayerNorm>(
        "temporal_linear.3", spatial_dim, layer_norm_eps, dtype, device);

    INFINICORE_NN_MODULE_INIT(mlp, spatial_dim, text_hidden_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(after_norm, text_hidden_size_, layer_norm_eps, dtype, device);
}

infinicore::Tensor Ernie4_5_VLResampler::forward(const infinicore::Tensor &x,
                                                 const infinicore::Tensor &grid_thw) const {
    // Input:  x [num_patches, pixel_hidden_size]
    // Output: [num_merged_tokens, text_hidden_size]
    ASSERT(x->ndim() == 2);
    size_t num_patches = x->shape()[0];
    size_t spatial_block = spatial_conv_size_ * spatial_conv_size_;
    size_t spatial_dim = pixel_hidden_size_ * spatial_block;

    ASSERT_EQ(num_patches % spatial_block, 0);
    size_t num_spatial_merged = num_patches / spatial_block;

    auto merged_spatial = x->view({num_spatial_merged, spatial_dim});

    // spatial_linear Sequential: Linear -> act -> Linear -> LayerNorm
    // TODO(ernie-vl): activation type (gelu vs quick_gelu) is a guess; verify.
    auto h = spatial_linear_0_->forward(merged_spatial);
    h = infinicore::op::gelu(h);
    h = spatial_linear_2_->forward(h);
    h = spatial_linear_3_->forward(h);
    auto x_spatial = h;

    // Detect video (any t > 1) on CPU. For images (t==1) we skip temporal merge.
    auto thw_cpu = grid_thw->to(infinicore::Device::cpu())->contiguous();
    const auto *thw_ptr = reinterpret_cast<const int64_t *>(thw_cpu->data());
    size_t num_media = grid_thw->shape()[0];
    bool has_video = false;
    for (size_t i = 0; i < num_media; ++i) {
        if (thw_ptr[i * 3] > 1) {
            has_video = true;
            break;
        }
    }

    infinicore::Tensor x_temporal;
    if (has_video) {
        ASSERT_EQ(num_spatial_merged % temporal_conv_size_, 0);
        size_t num_temporal_merged = num_spatial_merged / temporal_conv_size_;
        size_t temporal_dim = spatial_dim * temporal_conv_size_;
        auto x_temporal_in = x_spatial->view({num_temporal_merged, temporal_dim});

        // temporal_linear Sequential: Linear -> act -> Linear -> LayerNorm
        auto t = temporal_linear_0_->forward(x_temporal_in);
        t = infinicore::op::gelu(t);
        t = temporal_linear_2_->forward(t);
        t = temporal_linear_3_->forward(t);
        x_temporal = t;
    } else {
        x_temporal = x_spatial;
    }

    auto projected = mlp_->forward(x_temporal);  // -> [num_merged_tokens, text_hidden_size]
    return after_norm_->forward(projected);
}

} // namespace infinilm::models::ernie4_5_moe_vl
