#include "mask.cuh"

// Midterm Concepts:
// - 2D grid/thread mapping: row and col calculation
// - Scaling by 1.0f / sqrt(d_k)
// - Causal masking: lower triangular condition (col > row -> -1e9f)
// - Minimizing warp divergence: evaluate warp alignment or branchless selection (fminf / ternary)
__global__ void scale_and_mask_kernel(float *scores, int batch_size, int seq_len, float scale,
                                      bool causal) {
    // TODO: Your implementation here
}

void launch_scale_and_mask(float* d_scores, int batch_size, int seq_len,
                           float scale, bool causal, cudaStream_t stream) {
    // TODO: Determine gridDim and blockDim (e.g., 2D block (32, 8) or 1D mapping)
    // dim3 block(32, 8);
    // dim3 grid((seq_len + 31) / 32, (batch_size * seq_len + 7) / 8);
    // scale_and_mask_kernel<<<grid, block, 0, stream>>>(d_scores, batch_size, seq_len, scale, causal);
    // CHECK_LAST_CUDA_ERROR();
}
