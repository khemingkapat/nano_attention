#include "attention.cuh"
#include "test_utils.cuh"

bool run_case(const char *name, AttentionConfig cfg) {
    printf("\n=== %s (B=%d, H=%d, N=%d, D=%d, causal=%s) ===\n",
           name, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
           cfg.causal ? "true" : "false");

    int total_matrices = cfg.batch_size * cfg.num_heads;
    size_t q_size = (size_t)total_matrices * cfg.seq_len * cfg.head_dim;
    size_t workspace_size = (size_t)total_matrices * cfg.seq_len * cfg.seq_len;

    // 1. Host data
    auto h_Q = random_vector(q_size, 42);
    auto h_K = random_vector(q_size, 43);
    auto h_V = random_vector(q_size, 44);
    std::vector<float> h_ref(q_size);
    std::vector<float> h_out;

    // 2. CPU Reference (ground truth for all B x H matrices)
    for (int i = 0; i < total_matrices; ++i) {
        int offset = i * cfg.seq_len * cfg.head_dim;
        attention_cpu(h_Q.data() + offset,
                      h_K.data() + offset,
                      h_V.data() + offset,
                      h_ref.data() + offset,
                      cfg.seq_len, cfg.head_dim, cfg.causal);
    }

    // 3. GPU execution
    DeviceBuffer d_Q(h_Q);
    DeviceBuffer d_K(h_K);
    DeviceBuffer d_V(h_V);
    DeviceBuffer d_workspace(workspace_size);
    DeviceBuffer d_Out(q_size);

    NanoAttention attn(cfg);
    GpuTimer t;
    t.start();
    attn.forward_sync(d_Q, d_K, d_V, d_Out, d_workspace);
    t.stop();

    printf("  GPU Time   : %.3f ms\n", t.elapsed_ms());

    // 4. Verify against CPU
    d_Out.to_host(h_out);
    printf("  [Check]    ");
    return verify(h_ref, h_out, 1e-3f);
}

int main() {
    printf("=== Correctness Test: NanoAttention (Day 5 Synchronous Pipeline) ===\n");

    bool all_pass = true;

    // Case 1: Baseline power-of-2
    all_pass &= run_case("Baseline Single-Head",
                         AttentionConfig{1, 1, 64, 64, true});

    // Case 2: Multi-Batch & Multi-Head
    all_pass &= run_case("Multi-Batch Multi-Head",
                         AttentionConfig{2, 4, 64, 32, true});

    // Case 3: Non-power-of-2 (boundary guard check)
    all_pass &= run_case("Non-Power-of-2 Sequence",
                         AttentionConfig{1, 2, 47, 32, true});

    // Case 4: Non-causal (bidirectional)
    all_pass &= run_case("Non-Causal Attention",
                         AttentionConfig{1, 1, 64, 64, false});

    printf("\n=======================================================\n");
    if (all_pass) {
        printf("🎉 ALL ATTENTION TESTS PASSED!\n");
        return 0;
    } else {
        printf("❌ SOME ATTENTION TESTS FAILED!\n");
        return 1;
    }
}
