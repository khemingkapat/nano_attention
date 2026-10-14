#include "matmul.cuh"

// Day 1 Baseline: Naive GEMM Kernel
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int M, int K, int N,
                                    bool transpose_B) {
    // TODO (Student):
    // 1. Calculate global 2D thread coordinates: row and col
    // 2. Boundary check: ensure (row < M && col < N)
    // 3. Dot product accumulator over k = 0 ... K-1:
    //    - If transpose_B == true: B is (N, K), index is (col * K + k)
    //    - If transpose_B == false: B is (K, N), index is (k * N + col)
    // 4. Store result into C[row * N + col]
    //
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float b_val, sum;

    if (row < M && col < N) {
        sum = 0.0f;
        for (int k = 0; k < K; k++) {
            if (transpose_B) {
                b_val = B[col * K + k];
            } else {
                b_val = B[k * N + col];
            }
            sum += A[row * K + k] * b_val;
        }
        C[row * N + col] = sum;
    }
}

// Day 2: Tiled Matrix Multiplication
// Midterm Concepts:
// - __shared__ float s_A[TILE_SIZE][TILE_SIZE], s_B[TILE_SIZE][TILE_SIZE]
// - Double __syncthreads() (post-load barrier & post-compute barrier)
// - Boundary checking with zero padding for arbitrary sequence length N
// - Coalesced 128B DRAM bursts with (32, 8) or (16, 16) thread block layout
template <int TILE_WIDTH = 16>
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C, int M, int K, int N,
                                    bool transpose_B) {
    // TODO: Your implementation here
}

void launch_matmul(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                   bool transpose_B, cudaStream_t stream) {
    // TODO (Student):
    // 1. Define block dimensions (e.g. dim3 block(16, 16))
    // 2. Compute grid dimensions with the ceiling formula: (N + block.x - 1) / block.x, etc.
    // 3. Launch matmul_naive_kernel<<<grid, block, 0, stream>>>(...)
    // 4. CHECK_LAST_CUDA_ERROR();
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N, transpose_B);
    CHECK_LAST_CUDA_ERROR();
}
