// Day 1 / Day 2: matmul correctness. Only test cases live here —
// the kernel is in src/matmul.cu, helpers in include/test_utils.cuh.
#include "matmul.cuh"
#include "test_utils.cuh"

bool run_case(const char *name, int M, int K, int N, bool transpose_B) {
    printf("\n=== %s (M=%d K=%d N=%d T=%d) ===\n", name, M, K, N, transpose_B);

    auto h_A = random_vector((size_t)M * K, 1);
    auto h_B = random_vector((size_t)K * N, 2); // same count for [K,N] and [N,K]
    std::vector<float> h_ref((size_t)M * N), h_out;

    DeviceBuffer d_A(h_A), d_B(h_B), d_C((size_t)M * N);

    GpuTimer t;
    t.start();
    launch_matmul(d_A, d_B, d_C, M, K, N, transpose_B);
    t.stop();
    printf("  GPU time: %.3f ms\n", t.elapsed_ms());

    d_C.to_host(h_out);
    matmul_cpu(h_A.data(), h_B.data(), h_ref.data(), M, K, N, transpose_B);
    return verify(h_ref, h_out);
}

int main() {
    bool ok = true;
    ok &= run_case("A @ B", 64, 64, 64, false);
    ok &= run_case("Q @ K^T", 64, 64, 64, true);
    ok &= run_case("Odd dims (boundary guard)", 37, 23, 49, true);
    ok &= run_case("Odd dims, no transpose", 37, 23, 49, false);
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
