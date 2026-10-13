#pragma once

#include "common.cuh"

// Midterm Concepts:
// - Parallel Tree Reduction (Sequential addressing tid < s to eliminate intra-warp divergence)
// - Row-wise maximum finding for numerical stability: max_i (x_i)
// - Row-wise sum of exponentials: sum_i (exp(x_i - max))
// - Elementwise normalization: x_i = exp(x_i - max) / sum
// - Thread coarsening if sequence length > block size

__global__ void softmax_forward_kernel(float* scores, int batch_size, int seq_len) {
    // TODO: Your implementation here
}

void launch_softmax(float* d_scores, int batch_size, int seq_len, cudaStream_t stream = 0);
