#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,                  \
                    cudaGetErrorString(err_), cudaGetErrorName(err_));                             \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                                                    \
    do {                                                                                           \
        cudaError_t err_ = cudaGetLastError();                                                     \
        if (err_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s (%s)\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err_), cudaGetErrorName(err_));                             \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// ── Kernel Implementation (from src/matmul.cu) ─────────────────────────
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int M, int K, int N,
                                    bool transpose_B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float b_val = transpose_B ? B[col * K + k] : B[k * N + col];
            sum += A[row * K + k] * b_val;
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul(const float *d_A, const float *d_B, float *d_C, int M, int K, int N,
                   bool transpose_B, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N, transpose_B);
    CHECK_LAST_CUDA_ERROR();
}

// ── CPU References ──────────────────────────────────────────────────────
void matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N, bool transpose_B) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                double a_val = (double)A[i * K + k];
                double b_val = transpose_B ? (double)B[j * K + k] : (double)B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = (float)sum;
        }
    }
}

bool verify(const float *cpu, const float *gpu, int size, float tol = 1e-3f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(cpu[i] - gpu[i]);
        float denom = fabsf(cpu[i]) + 1e-6f;
        float rel = diff / denom;
        if (diff > max_abs_err)
            max_abs_err = diff;
        if (rel > max_rel_err) {
            max_rel_err = rel;
            worst_idx = i;
        }
    }
    bool pass = (max_abs_err < 1e-3f) || (max_rel_err < tol);
    printf("  Max abs error: %e | Max rel error: %e (worst cell [%d]: cpu=%.6f, gpu=%.6f) -> %s\n",
           max_abs_err, max_rel_err, worst_idx, cpu[worst_idx], gpu[worst_idx],
           pass ? "PASS [OK]" : "FAIL [X]");
    return pass;
}

// ── Test Runner ─────────────────────────────────────────────────────────
void run_test_case(const char *test_name, int M, int K, int N, bool transpose_B) {
    printf("\n=== Running Test: %s (M=%d, K=%d, N=%d, transpose_B=%d) ===\n", test_name, M, K, N,
           transpose_B);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = (transpose_B ? (N * K) : (K * N)) * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C_cpu = (float *)malloc(bytes_C);
    float *h_C_gpu = (float *)malloc(bytes_C);

    srand(42);
    int num_elements_B = transpose_B ? (N * K) : (K * N);
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < num_elements_B; ++i)
        h_B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Time GPU Kernel with cudaEvent
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    launch_matmul(d_A, d_B, d_C, M, K, N, transpose_B);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  GPU Kernel Time: %.3f ms\n", ms);

    // Copy back
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Compute CPU reference
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N, transpose_B);

    // Verify
    verify(h_C_cpu, h_C_gpu, M * N);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
}

int main() {
    printf("##################################################\n");
    printf("#  nano_attention: Day 1 Naive GEMM Verification  #\n");
    printf("##################################################\n");

    // Test 1: Standard Matrix Multiplication (e.g., Scores x V)
    run_test_case("Standard GEMM (A @ B)", 64, 64, 64, false);

    // Test 2: Attention Q @ K^T (transposed B)
    // Q is (M x d_k) = (64 x 64), K is (N x d_k) = (64 x 64)
    run_test_case("Attention Scores (Q @ K^T)", 64, 64, 64, true);

    // Test 3: Arbitrary / Non-multiple of 16 dimensions (Boundary Guard Test)
    run_test_case("Boundary Guard Check (Odd Dimensions)", 37, 23, 49, true);

    printf("\nAll Day 1 checks complete!\n");
    return 0;
}
