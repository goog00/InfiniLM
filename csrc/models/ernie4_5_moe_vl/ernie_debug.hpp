// TEMPORARY DEBUG HELPER -- remove before final submission.
// Prints hidden-state tensor stats when env var ERNIE_DBG is set, to localize
// where the text forward pass loses information. Gated by getenv so it is a
// no-op in normal runs.
#pragma once

#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace infinilm::models::ernie4_5_moe_vl {

inline void ernie_dbg_stats(const char *tag, const infinicore::Tensor &t) {
    static const bool on = (std::getenv("ERNIE_DBG") != nullptr);
    if (!on) {
        return;
    }
    auto c = t->to(infinicore::Device::cpu())->contiguous();
    size_t n = c->numel();
    if (n == 0) {
        std::fprintf(stderr, "[ERNIE_DBG] %s numel=0\n", tag);
        return;
    }
    auto dtype = c->dtype();
    const void *raw = c->data();
    auto rd = [&](size_t i) -> float {
        if (dtype == infinicore::DataType::BF16) {
            return bf16_to_f32(reinterpret_cast<const uint16_t *>(raw)[i]);
        } else if (dtype == infinicore::DataType::F16) {
            return f16_to_f32(reinterpret_cast<const uint16_t *>(raw)[i]);
        }
        return reinterpret_cast<const float *>(raw)[i];
    };
    float mn = rd(0), mx = rd(0), sum = 0.f, absmax = 0.f;
    bool nan = false, inf = false;
    for (size_t i = 0; i < n; ++i) {
        float v = rd(i);
        if (std::isnan(v)) nan = true;
        if (std::isinf(v)) inf = true;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        absmax = std::max(absmax, std::fabs(v));
    }
    // Also print the first few values of the last row, to check per-position variation.
    std::fprintf(stderr,
                 "[ERNIE_DBG] %-16s numel=%zu min=%.4f max=%.4f mean=%.6f absmax=%.4f nan=%d inf=%d | head=%.4f %.4f %.4f tail=%.4f %.4f %.4f\n",
                 tag, n, mn, mx, sum / static_cast<float>(n), absmax, int(nan), int(inf),
                 rd(0), rd(1 % n), rd(2 % n),
                 rd(n - 3), rd(n - 2), rd(n - 1));
    std::fflush(stderr);
}

} // namespace infinilm::models::ernie4_5_moe_vl
