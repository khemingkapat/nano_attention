#pragma once
// PUBLIC INTERFACE ONLY. Kernels live in src/matmul.cu (they are private details).

#include "common.cuh"

// C[M,N] = A[M,K] @ B,  B is [K,N] (or [N,K] when transpose_B, i.e. Q @ K^T)
void launch_matmul(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                   bool transpose_B = false, cudaStream_t stream = 0);
void launch_matmul_naive(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                         bool transpose_B = false, cudaStream_t stream = 0);
void launch_matmul_tiled(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                         bool transpose_B = false, cudaStream_t stream = 0);

