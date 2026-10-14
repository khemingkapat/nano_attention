#pragma once

#include "common.cuh"

// Day 1 Baseline: Naive GEMM Kernel
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N,
                                    bool transpose_B = false);

// Day 2: Tiled Matrix Multiplication
// Midterm Concepts:
// - __shared__ float s_A[TILE_SIZE][TILE_SIZE], s_B[TILE_SIZE][TILE_SIZE]
// - Double __syncthreads() (post-load barrier & post-compute barrier)
// - Boundary checking with zero padding for arbitrary sequence length N
// - Coalesced 128B DRAM bursts with (32, 8) or (16, 16) thread block layout
template <int TILE_WIDTH = 16>
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N,
                                    bool transpose_B = false) {
    // TODO: Your implementation here
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                   int M, int K, int N, bool transpose_B = false,
                   cudaStream_t stream = 0);
