#pragma once

#include "common.cuh"

// Midterm Concepts:
// - 2D grid/thread mapping: row and col calculation
// - Scaling by 1.0f / sqrt(d_k)
// - Causal masking: lower triangular condition (col > row -> -1e9f)
// - Minimizing warp divergence: evaluate warp alignment or branchless selection (fminf / ternary)

__global__ void scale_and_mask_kernel(float* scores, int batch_size, int seq_len, float scale, bool causal) {
    // TODO: Your implementation here
}

void launch_scale_and_mask(float* d_scores, int batch_size, int seq_len,
                           float scale, bool causal = true, cudaStream_t stream = 0);
