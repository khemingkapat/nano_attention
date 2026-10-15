#include "matmul.cuh"

// Day 1 Baseline: Naive GEMM Kernel
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int M, int K, int N,
                                    bool transpose_B) {
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
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;

    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_WIDTH + tx;
        if (row < M && a_col < K) {
            s_A[ty][tx] = A[row * K + a_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_WIDTH + ty;
        if (b_row < K && col < N) {
            if (transpose_B) {
                s_B[ty][tx] = B[col * K + b_row];
            } else {
                s_B[ty][tx] = B[b_row * N + col];
            }
        } else {
            s_B[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                         bool transpose_B, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N, transpose_B);
    CHECK_LAST_CUDA_ERROR();
}

void launch_matmul_tiled(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                         bool transpose_B, cudaStream_t stream) {
    constexpr int TILE_WIDTH = 16;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_tiled_kernel<TILE_WIDTH><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N, transpose_B);
    CHECK_LAST_CUDA_ERROR();
}

void launch_matmul(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                   bool transpose_B, cudaStream_t stream) {
    launch_matmul_tiled(d_A, d_B, d_C, M, K, N, transpose_B, stream);
}

