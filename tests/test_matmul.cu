// Day 1 / Day 2: matmul correctness. Only test cases live here —
// the kernel is in src/matmul.cu, helpers in include/test_utils.cuh.
#include "matmul.cuh"
#include "test_utils.cuh"

bool run_case(const char *name, int M, int K, int N, bool transpose_B) {
    printf("\n=== %s (M=%d K=%d N=%d T=%d) ===\n", name, M, K, N, transpose_B);

    auto h_A = random_vector((size_t)M * K, 1);
    auto h_B = random_vector((size_t)K * N, 2); // same count for [K,N] and [N,K]
    std::vector<float> h_ref((size_t)M * N), h_out_naive, h_out_tiled;

    DeviceBuffer d_A(h_A), d_B(h_B);
    DeviceBuffer d_C_naive((size_t)M * N);
    DeviceBuffer d_C_tiled((size_t)M * N);

    GpuTimer t;

    // 1. Time Naive GEMM
    t.start();
    launch_matmul_naive(d_A, d_B, d_C_naive, M, K, N, transpose_B);
    t.stop();
    float time_naive = t.elapsed_ms();

    // 2. Time Tiled GEMM
    t.start();
    launch_matmul_tiled(d_A, d_B, d_C_tiled, M, K, N, transpose_B);
    t.stop();
    float time_tiled = t.elapsed_ms();

    printf("  Naive GEMM : %.3f ms\n", time_naive);
    printf("  Tiled GEMM : %.3f ms\n", time_tiled);
    if (time_tiled > 0.0f) {
        printf("  Speedup    : %.2fx\n", time_naive / time_tiled);
    }

    // 3. Verify correctness of both against CPU
    matmul_cpu(h_A.data(), h_B.data(), h_ref.data(), M, K, N, transpose_B);

    d_C_naive.to_host(h_out_naive);
    printf("  [Naive Check] ");
    bool ok_naive = verify(h_ref, h_out_naive);

    d_C_tiled.to_host(h_out_tiled);
    printf("  [Tiled Check] ");
    bool ok_tiled = verify(h_ref, h_out_tiled);

    return ok_naive && ok_tiled;
}

int main() {
    bool ok = true;
    ok &= run_case("A @ B (Small)", 64, 64, 64, false);
    ok &= run_case("Q @ K^T (Small)", 64, 64, 64, true);
    ok &= run_case("Odd dims (boundary guard)", 37, 23, 49, true);
    ok &= run_case("Odd dims, no transpose", 37, 23, 49, false);
    ok &= run_case("A @ B (Medium 512x512)", 512, 512, 512, false);
    ok &= run_case("Q @ K^T (Attention 1024x64)", 1024, 64, 1024, true);
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
