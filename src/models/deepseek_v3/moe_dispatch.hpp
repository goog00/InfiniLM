#pragma once

#include <cstdint>

extern "C" {
// Build per-expert positions from top-k indices.
// indices: device pointer of int32, length = ntok * topk
// expert_counts: device pointer of int32, length = nexperts (output, zero-initialized)
// expert_pos: device pointer of int32, length = nexperts * capacity (output)
void build_expert_index_gpu(const int32_t *indices, int ntok, int topk, int nexperts, int capacity,
                           int32_t *expert_counts, int32_t *expert_pos);

// Gather hidden states for one expert into contiguous buffer
// hidden: device ptr float, shape (ntok, hidden_dim)
// positions: device ptr int32, length = count
// out: device ptr float, shape (count, hidden_dim)
void gather_tokens_gpu(const float *hidden, int ntok, int hidden_dim,
                       const int32_t *positions, int count, float *out);

// Scatter-add expert outputs back to router_states_sum
// out: device ptr float, shape (count, hidden_dim)
// positions: device ptr int32, length = count
// router_states_sum: device ptr float, shape (ntok, hidden_dim)
void scatter_add_tokens_gpu(const float *out, int count, int hidden_dim,
                           const int32_t *positions, float *router_states_sum);
}
