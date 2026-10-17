#include "attention.cuh"
#include "test_utils.cuh"

bool run_case(const char *name, AttentionConfig cfg) {
    printf("\n=== %s (B=%d, H=%d, N=%d, D=%d, causal=%s) ===\n",
           name, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
           cfg.causal ? "true" : "false");

    int total_matrices = cfg.batch_size * cfg.num_heads;
    size_t q_size = (size_t)total_matrices * cfg.seq_len * cfg.head_dim;
    size_t workspace_size = (size_t)total_matrices * cfg.seq_len * cfg.seq_len;

    // 1. Pinned Host Data (Enables true async DMA)
    PinnedBuffer<float> h_Q(q_size);
    PinnedBuffer<float> h_K(q_size);
    PinnedBuffer<float> h_V(q_size);
    PinnedBuffer<float> h_out_sync(q_size);
    PinnedBuffer<float> h_stream_out(q_size);

    h_Q.fill_random(42);
    h_K.fill_random(43);
    h_V.fill_random(44);

    std::vector<float> h_ref(q_size);

    // 2. CPU Reference (ground truth for all B x H matrices)
    for (int i = 0; i < total_matrices; ++i) {
        int offset = i * cfg.seq_len * cfg.head_dim;
        attention_cpu(h_Q.data() + offset,
                      h_K.data() + offset,
                      h_V.data() + offset,
                      h_ref.data() + offset,
                      cfg.seq_len, cfg.head_dim, cfg.causal);
    }

    // 3. Device buffers for Sync baseline
    DeviceBuffer d_Q(q_size);
    DeviceBuffer d_K(q_size);
    DeviceBuffer d_V(q_size);
    DeviceBuffer d_workspace(workspace_size);
    DeviceBuffer d_Out(q_size);

    NanoAttention attn(cfg);
    size_t q_bytes = q_size * sizeof(float);

    // --- FAIR MEASUREMENT 1: Full Synchronous End-to-End (H2D + Compute + D2H) ---
    GpuTimer t_sync;
    t_sync.start();
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), q_bytes, cudaMemcpyHostToDevice));
    attn.forward_sync(d_Q, d_K, d_V, d_Out, d_workspace);
    CHECK_CUDA(cudaMemcpy(h_out_sync.data(), d_Out, q_bytes, cudaMemcpyDeviceToHost));
    t_sync.stop();

    // --- FAIR MEASUREMENT 2: Streamed (2 streams) ---
    GpuTimer t_stream2;
    t_stream2.start();
    attn.forward_streamed(h_Q.data(), h_K.data(), h_V.data(), h_stream_out.data(), /*num_streams=*/2);
    t_stream2.stop();

    // --- FAIR MEASUREMENT 3: Streamed (4 streams) ---
    PinnedBuffer<float> h_stream_out4(q_size);
    GpuTimer t_stream4;
    t_stream4.start();
    attn.forward_streamed(h_Q.data(), h_K.data(), h_V.data(), h_stream_out4.data(), /*num_streams=*/4);
    t_stream4.stop();

    printf("  Sync End-to-End       : %.3f ms\n", t_sync.elapsed_ms());
    printf("  Streamed (2 streams)  : %.3f ms\n", t_stream2.elapsed_ms());
    printf("  Streamed (4 streams)  : %.3f ms\n", t_stream4.elapsed_ms());

    // 4. Verify against CPU reference
    std::vector<float> v_sync(h_out_sync.data(), h_out_sync.data() + q_size);
    std::vector<float> v_stream2(h_stream_out.data(), h_stream_out.data() + q_size);
    std::vector<float> v_stream4(h_stream_out4.data(), h_stream_out4.data() + q_size);

    printf("  [Sync Check]       ");
    bool pass_sync = verify(h_ref, v_sync, 1e-3f);
    printf("  [Stream-2 Check]   ");
    bool pass_s2 = verify(h_ref, v_stream2, 1e-3f);
    printf("  [Stream-4 Check]   ");
    bool pass_s4 = verify(h_ref, v_stream4, 1e-3f);

    return pass_sync && pass_s2 && pass_s4;
}

int main() {
    printf("=== Fair End-to-End Benchmark & Test: Sync vs Streamed ===\n");

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

    // Case 5: Realistic Transformer Workload (B=4, H=8, N=128, D=64)
    all_pass &= run_case("Large Multi-Head (B=4, H=8, N=128, D=64)",
                         AttentionConfig{4, 8, 128, 64, true});

    printf("\n=======================================================\n");
    if (all_pass) {
        printf("🎉 ALL ATTENTION TESTS PASSED!\n");
        return 0;
    } else {
        printf("❌ SOME ATTENTION TESTS FAILED!\n");
        return 1;
    }
}
