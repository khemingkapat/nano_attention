#include "softmax.cuh"
#include <cfloat>

// ─────────────────────────────────────────────────────────────────────────────
// Midterm Concepts:
// 1. Numerical stability: subtract row max m to prevent FP32 exp overflow.
// 2. Sequential Addressing reduction: (tid < s) eliminates intra-warp divergence
//    until s < 32.
// 3. Shared memory synchronization: __syncthreads() after loading & each stride.
// 4. Identity element: -FLT_MAX (or -1e9f) for max reduction, 0.0f for sum reduction.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void softmax_forward_kernel(float *scores, int total_rows, int seq_len) {
    // Dynamically allocated shared memory (size specified in <<<grid, block, shmem_bytes>>>)
    extern __shared__ float s_data[];

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= total_rows)
        return;

    float *row = scores + (size_t)row_idx * seq_len;

    // ── Phase 1: Find Row Maximum (Tree Reduction) ───────────────────────────
    // 1. Cooperative load into shared memory (with out-of-bounds padding):
    s_data[tid] = (tid < seq_len) ? row[tid] : -FLT_MAX;
    __syncthreads();

    // 2. Sequential addressing tree reduction for MAX:
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    // 3. Broadcast max value:
    float row_max = s_data[0];
    __syncthreads();

    // ── Phase 2: Exponentiate and Sum (Tree Reduction) ────────────────────────
    // 4. Compute exp(x - max) and load into shared memory:
    float my_exp = (tid < seq_len) ? expf(row[tid] - row_max) : 0.0f;
    s_data[tid] = my_exp;
    __syncthreads();

    // 5. Sequential addressing tree reduction for SUM:
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = s_data[tid] + s_data[tid + s];
        }
        __syncthreads();
    }

    // 6. Broadcast sum value:
    float row_sum = s_data[0];
    __syncthreads();

    // ── Phase 3: Normalize and Write Back ────────────────────────────────────
    // 7. Write normalized probability to row[tid]:
    if (tid < seq_len) {
        row[tid] = my_exp / row_sum;
    }
}

void launch_softmax(float *d_scores, int batch_size, int seq_len, cudaStream_t stream) {
    int total_rows = batch_size * seq_len;

    // Find the next power-of-2 >= seq_len (at least 32, max 1024)
    int block_size = 32;
    while (block_size < seq_len && block_size < 1024) {
        block_size *= 2;
    }

    dim3 grid(total_rows);
    dim3 block(block_size);
    size_t shmem_bytes = block_size * sizeof(float);

    softmax_forward_kernel<<<grid, block, shmem_bytes, stream>>>(d_scores, total_rows, seq_len);
    CHECK_LAST_CUDA_ERROR();
}
