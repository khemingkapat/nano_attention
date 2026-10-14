#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// test_utils.cuh — HOST-ONLY helpers shared by every test/benchmark.
//   * random fill, verification, CPU reference implementations
//   * DeviceBuffer: RAII wrapper so tests don't repeat cudaMalloc/cudaFree
// Nothing in here is exam material; it exists so test files stay ~30 lines.
// All functions are `inline` because this header is included, not compiled.
// ─────────────────────────────────────────────────────────────────────────────

#include "common.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ── Device memory RAII ──────────────────────────────────────────────────────
// Usage:  DeviceBuffer d_A(h_A);          // cudaMalloc + H2D copy
//         DeviceBuffer d_C(M * N);        // cudaMalloc only
//         d_C.to_host(h_C);               // D2H copy
//         (cudaFree happens automatically at end of scope)
struct DeviceBuffer {
    float *ptr = nullptr;
    size_t count = 0;

    explicit DeviceBuffer(size_t n) : count(n) {
        CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float)));
    }
    explicit DeviceBuffer(const std::vector<float> &host) : DeviceBuffer(host.size()) {
        CHECK_CUDA(cudaMemcpy(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void to_host(std::vector<float> &host) const {
        host.resize(count);
        CHECK_CUDA(cudaMemcpy(host.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    operator float *() const { return ptr; } // pass DeviceBuffer straight to launch_*()
};

// ── Data generation ─────────────────────────────────────────────────────────
inline std::vector<float> random_vector(size_t n, unsigned seed = 42) {
    std::vector<float> v(n);
    srand(seed);
    for (auto &x : v)
        x = (rand() / (float)RAND_MAX) * 2.0f - 1.0f; // uniform [-1, 1]
    return v;
}

// ── Verification ────────────────────────────────────────────────────────────
inline bool verify(const std::vector<float> &cpu, const std::vector<float> &gpu,
                   float tol = 1e-3f) {
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        float diff = fabsf(cpu[i] - gpu[i]);
        float rel = diff / (fabsf(cpu[i]) + 1e-6f);
        if (diff > max_abs)
            max_abs = diff;
        if (rel > max_rel) {
            max_rel = rel;
            worst = i;
        }
    }
    bool pass = (max_abs < 1e-3f) || (max_rel < tol);
    printf("  max_abs=%.2e  max_rel=%.2e  (worst [%zu]: cpu=%.6f gpu=%.6f)  -> %s\n", max_abs,
           max_rel, worst, cpu[worst], gpu[worst], pass ? "PASS" : "FAIL");
    return pass;
}

// ── CPU references (double accumulation = ground truth) ─────────────────────
// C[M,N] = A[M,K] @ B   where B is [K,N], or [N,K] if transpose_B
inline void matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N,
                       bool transpose_B) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += (double)A[i * K + k] * (double)(transpose_B ? B[j * K + k] : B[k * N + j]);
            C[i * N + j] = (float)sum;
        }
}

// In-place: scores[rows, seq_len] *= scale, then mask col > row (within each seq_len x seq_len)
inline void scale_and_mask_cpu(float *scores, int batch_size, int seq_len, float scale,
                               bool causal) {
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < seq_len; ++j) {
                float &s = scores[(b * seq_len + i) * seq_len + j];
                s = (causal && j > i) ? -1e9f : s * scale;
            }
}

// In-place row-wise softmax over [num_rows, row_len]
inline void softmax_cpu(float *x, int num_rows, int row_len) {
    for (int r = 0; r < num_rows; ++r) {
        float *row = x + (size_t)r * row_len;
        float m = row[0];
        for (int j = 1; j < row_len; ++j)
            m = fmaxf(m, row[j]);
        double sum = 0.0;
        for (int j = 0; j < row_len; ++j) {
            row[j] = expf(row[j] - m);
            sum += row[j];
        }
        for (int j = 0; j < row_len; ++j)
            row[j] = (float)(row[j] / sum);
    }
}

// Single-head attention: Out[S,D] = softmax(Q K^T / sqrt(D) [+mask]) V
inline void attention_cpu(const float *Q, const float *K, const float *V, float *Out, int seq_len,
                          int head_dim, bool causal) {
    std::vector<float> scores((size_t)seq_len * seq_len);
    matmul_cpu(Q, K, scores.data(), seq_len, head_dim, seq_len, /*transpose_B=*/true);
    scale_and_mask_cpu(scores.data(), 1, seq_len, 1.0f / sqrtf((float)head_dim), causal);
    softmax_cpu(scores.data(), seq_len, seq_len);
    matmul_cpu(scores.data(), V, Out, seq_len, seq_len, head_dim, /*transpose_B=*/false);
}
