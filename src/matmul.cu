#include "matmul.cuh"

void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                   int M, int K, int N, bool transpose_B,
                   cudaStream_t stream) {
    // TODO: Determine gridDim and blockDim
    // Example: dim3 block(16, 16);
    //          dim3 grid((N + 15) / 16, (M + 15) / 16);
    // matmul_tiled_kernel<16><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N, transpose_B);
    // CHECK_LAST_CUDA_ERROR();
}
