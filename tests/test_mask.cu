// Day 3: Scale and Mask correctness & timing test.
#include "mask.cuh"
#include "test_utils.cuh"

bool run_case(const char *name, int batch_size, int seq_len, float scale, bool causal) {
    printf("\n=== %s (B=%d S=%d scale=%.4f causal=%d) ===\n", name, batch_size, seq_len, scale,
           causal);

    size_t total_elements = (size_t)batch_size * seq_len * seq_len;
    auto h_in = random_vector(total_elements, 42);
    std::vector<float> h_ref = h_in;
    std::vector<float> h_out;

    // 1. CPU Reference Ground Truth
    scale_and_mask_cpu(h_ref.data(), batch_size, seq_len, scale, causal);

    // 2. GPU In-Place Kernel
    DeviceBuffer d_scores(h_in);
    GpuTimer t;

    t.start();
    launch_scale_and_mask(d_scores, batch_size, seq_len, scale, causal);
    t.stop();

    float elapsed = t.elapsed_ms();
    printf("  GPU Time   : %.3f ms\n", elapsed);

    // 3. Verify correctness
    d_scores.to_host(h_out);
    printf("  [Check]    ");
    bool ok = verify(h_ref, h_out);

    return ok;
}

int main() {
    bool ok = true;
    // Standard causal self-attention (d_k = 64 -> scale = 1 / sqrt(64) = 0.125)
    ok &= run_case("Standard Causal (B=2, S=64)", 2, 64, 0.125f, true);

    // Non-causal attention (e.g., BERT / encoder)
    ok &= run_case("Non-Causal (B=2, S=64)", 2, 64, 0.125f, false);

    // Boundary guard check (odd dimensions not multiples of 32 or 8)
    ok &= run_case("Odd Dims Boundary Guard (B=3, S=37)", 3, 37, 0.164f, true);

    // Single sequence
    ok &= run_case("Single Batch (B=1, S=128)", 1, 128, 0.088f, true);

    // Larger workload
    ok &= run_case("Larger Context (B=4, S=512)", 4, 512, 0.044f, true);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
