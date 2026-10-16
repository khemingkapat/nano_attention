// Day 4: Row-wise Softmax with Tree Reduction correctness & stability test.
#include "softmax.cuh"
#include "test_utils.cuh"

bool run_case(const char *name, int batch_size, int seq_len, bool inject_large_values = false) {
    printf("\n=== %s (B=%d S=%d) ===\n", name, batch_size, seq_len);

    int total_rows = batch_size * seq_len;
    size_t total_elements = (size_t)total_rows * seq_len;
    auto h_in = random_vector(total_elements, 42);

    if (inject_large_values) {
        // Inject large values to test numerical stability (exp(100.0f) would overflow to inf without max subtraction)
        for (size_t i = 0; i < total_elements; i += seq_len) {
            h_in[i] = 100.0f;
            if (seq_len > 1) h_in[i + 1] = 95.0f;
        }
    }

    std::vector<float> h_ref = h_in;
    std::vector<float> h_out;

    // 1. CPU Reference Ground Truth
    softmax_cpu(h_ref.data(), total_rows, seq_len);

    // 2. GPU In-Place Kernel
    DeviceBuffer d_scores(h_in);
    GpuTimer t;

    t.start();
    launch_softmax(d_scores, batch_size, seq_len);
    t.stop();

    float elapsed = t.elapsed_ms();
    printf("  GPU Time   : %.3f ms\n", elapsed);

    // 3. Verify correctness
    d_scores.to_host(h_out);
    printf("  [Check]    ");
    bool ok = verify(h_ref, h_out, 1e-4f);

    // Check row sum ~= 1.0f and no NaN / Inf
    bool sum_ok = true;
    for (int r = 0; r < total_rows; ++r) {
        double row_sum = 0.0;
        for (int c = 0; c < seq_len; ++c) {
            float val = h_out[r * seq_len + c];
            if (std::isnan(val) || std::isinf(val)) {
                printf("  [ERROR] NaN or Inf detected at row %d col %d!\n", r, c);
                sum_ok = false;
                break;
            }
            row_sum += val;
        }
        if (fabs(row_sum - 1.0) > 1e-3) {
            printf("  [ERROR] Row %d sum is %.6f (expected 1.0)!\n", r, row_sum);
            sum_ok = false;
            break;
        }
        if (!sum_ok) break;
    }

    if (sum_ok) {
        printf("  [Stability] Row sums equal 1.0, no NaN/Inf detected.\n");
    }

    return ok && sum_ok;
}

int main() {
    bool ok = true;

    // Case 1: Standard size (power of 2)
    ok &= run_case("Standard Power of 2 (B=2, S=64)", 2, 64);

    // Case 2: Boundary guard test (odd dimension not a power of 2)
    ok &= run_case("Odd Sequence Length (B=3, S=37)", 3, 37);

    // Case 3: Single sequence
    ok &= run_case("Single Batch (B=1, S=128)", 1, 128);

    // Case 4: Larger context length
    ok &= run_case("Larger Context (B=4, S=256)", 4, 256);

    // Case 5: Numerical stability with large logits
    ok &= run_case("Numerical Stability Test (Large Logits: 100.0f)", 2, 64, /*inject_large_values=*/true);

    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
