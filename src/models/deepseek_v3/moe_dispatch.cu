#include "moe_dispatch.hpp"
#include <cuda_runtime.h>
#include <cstring>

__global__ void build_index_kernel(const int32_t *indices, int total, int topk, int nexperts, int capacity, int32_t *expert_counts, int32_t *expert_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int token = idx / topk;
    int expert = indices[idx];
    if (expert < 0 || expert >= nexperts) return;
    int pos = atomicAdd(&expert_counts[expert], 1);
    if (pos < capacity) {
        expert_pos[expert * capacity + pos] = token;
    }
}

extern "C" void build_expert_index_gpu(const int32_t *indices, int ntok, int topk, int nexperts, int capacity,
                                       int32_t *expert_counts, int32_t *expert_pos) {
    int total = ntok * topk;
    // zero counts
    cudaMemset(expert_counts, 0, sizeof(int32_t) * nexperts);
    // optional: fill expert_pos with -1
    cudaMemset(expert_pos, 0xff, sizeof(int32_t) * nexperts * capacity);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    build_index_kernel<<<blocks, threads>>>(indices, total, topk, nexperts, capacity, expert_counts, expert_pos);
    cudaDeviceSynchronize();
}

__global__ void gather_kernel(const float *hidden, int hidden_dim, const int32_t *positions, int count, float *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * hidden_dim;
    if (tid >= total) return;
    int token_idx = tid / hidden_dim;
    int dim = tid % hidden_dim;
    int token = positions[token_idx];
    out[tid] = hidden[token * hidden_dim + dim];
}

extern "C" void gather_tokens_gpu(const float *hidden, int ntok, int hidden_dim,
                                  const int32_t *positions, int count, float *out) {
    if (count <= 0) return;
    int total = count * hidden_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gather_kernel<<<blocks, threads>>>(hidden, hidden_dim, positions, count, out);
    cudaDeviceSynchronize();
}

__global__ void scatter_add_kernel(const float *out, int hidden_dim, const int32_t *positions, int count, float *router) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * hidden_dim;
    if (tid >= total) return;
    int token_idx = tid / hidden_dim;
    int dim = tid % hidden_dim;
    int token = positions[token_idx];
    atomicAdd(&router[token * hidden_dim + dim], out[tid]);
}

extern "C" void scatter_add_tokens_gpu(const float *out, int count, int hidden_dim,
                                      const int32_t *positions, float *router_states_sum) {
    if (count <= 0) return;
    int total = count * hidden_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_add_kernel<<<blocks, threads>>>(out, hidden_dim, positions, count, router_states_sum);
    cudaDeviceSynchronize();
}
