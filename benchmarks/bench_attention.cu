#include "attention.cuh"
#include "test_utils.cuh"
#include <iomanip>
#include <iostream>
#include <vector>

void print_device_info() {
    int device_id = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "========================================================================\n";
    std::cout << "                          GPU HARDWARE PROFILE                          \n";
    std::cout << "========================================================================\n";
    std::cout << "Device Name               : " << prop.name << "\n";
    std::cout << "Compute Capability        : " << prop.major << "." << prop.minor << "\n";
    std::cout << "SM Count (Multiprocessors): " << prop.multiProcessorCount << "\n";
    std::cout << "Warp Size                 : " << prop.warpSize << " threads\n";
    std::cout << "Shared Memory per Block   : " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "Registers per Block       : " << prop.regsPerBlock << "\n";
    std::cout << "Total Global Memory       : " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";

    // --- Theoretical Peak Memory Bandwidth ---
    // In modern CUDA, query device attributes:
    int memory_clock_khz = 0;
    int bus_width_bits = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device_id));

    // Clock rate is in kHz -> convert to Hz (* 1e3)
    // Bus width is in bits -> convert to bytes (/ 8.0)
    // DDR factor = 2.0
    double mem_clock_hz = (double)memory_clock_khz * 1e3;
    double bus_width_bytes = (double)bus_width_bits / 8.0;
    double peak_bandwidth_GBs = (2.0 * mem_clock_hz * bus_width_bytes) / 1e9;

    std::cout << "Memory Bus Width          : " << bus_width_bits << " bits\n";
    std::cout << "Memory Clock Rate         : " << memory_clock_khz / 1e3 << " MHz\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Theoretical Peak BW       : " << peak_bandwidth_GBs << " GB/s\n";
    std::cout << "========================================================================\n\n";
}

struct BenchMetrics {
    double flops;
    double dram_bytes;
    double dma_bytes;
    float time_sync_ms;
    float time_stream2_ms;
    float time_stream4_ms;

    double tflops_sync() const { return (flops / (time_sync_ms * 1e-3)) / 1e12; }
    double tflops_stream4() const { return (flops / (time_stream4_ms * 1e-3)) / 1e12; }
    double eff_bw_GBs_sync() const { return (dram_bytes / (time_sync_ms * 1e-3)) / 1e9; }
    double eff_bw_GBs_stream4() const { return (dram_bytes / (time_stream4_ms * 1e-3)) / 1e9; }
    double arithmetic_intensity() const { return flops / dram_bytes; }
    float speedup() const { return time_sync_ms / time_stream4_ms; }
};

BenchMetrics run_attention_benchmark(const char* label, AttentionConfig cfg, int num_iters = 30) {
    int total_matrices = cfg.batch_size * cfg.num_heads;
    size_t q_size = (size_t)total_matrices * cfg.seq_len * cfg.head_dim;
    size_t workspace_size = (size_t)total_matrices * cfg.seq_len * cfg.seq_len;
    size_t q_bytes = q_size * sizeof(float);

    // Host pinned buffers
    PinnedBuffer<float> h_Q(q_size);
    PinnedBuffer<float> h_K(q_size);
    PinnedBuffer<float> h_V(q_size);
    PinnedBuffer<float> h_Out_sync(q_size);
    PinnedBuffer<float> h_Out_s2(q_size);
    PinnedBuffer<float> h_Out_s4(q_size);

    h_Q.fill_random(101);
    h_K.fill_random(102);
    h_V.fill_random(103);

    // Device buffers for synchronous baseline
    DeviceBuffer d_Q(q_size);
    DeviceBuffer d_K(q_size);
    DeviceBuffer d_V(q_size);
    DeviceBuffer d_Out(q_size);
    DeviceBuffer d_workspace(workspace_size);

    NanoAttention attn(cfg);

    // ── Midterm Formulas: FLOPs & Memory I/O Traffic ─────────────────────────
    // 1. Q @ K^T: 2 * M * N * N * D FLOPs
    // 2. Scale & Mask: 2 * M * N * N FLOPs
    // 3. Softmax: 3 * M * N * N FLOPs (row max, exp+sum, div)
    // 4. P @ V: 2 * M * N * D * N FLOPs
    double M = (double)total_matrices;
    double N = (double)cfg.seq_len;
    double D = (double)cfg.head_dim;
    double total_flops = 4.0 * M * N * N * D + 5.0 * M * N * N;

    // Minimum DRAM traffic across kernel sequence (FP32 = 4 bytes):
    // QK^T reads Q, K (2*M*N*D*4), writes S (M*N*N*4)
    // Scale/Mask in-place: read & write S (2*M*N*N*4)
    // Softmax in-place: read & write S (2*M*N*N*4)
    // PV reads P (M*N*N*4), reads V (M*N*D*4), writes Out (M*N*D*4)
    double dram_bytes = M * (4.0 * N * D + 6.0 * N * N) * sizeof(float);

    // End-to-end DMA bytes (H2D Q, K, V + D2H Out)
    double dma_bytes = 4.0 * q_size * sizeof(float);

    // ── Warm-up (10 runs) ───────────────────────────────────────────────────
    for (int i = 0; i < 10; ++i) {
        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), q_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), q_bytes, cudaMemcpyHostToDevice));
        attn.forward_sync(d_Q, d_K, d_V, d_Out, d_workspace);
        CHECK_CUDA(cudaMemcpy(h_Out_sync.data(), d_Out, q_bytes, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // ── Benchmark: Synchronous End-to-End ───────────────────────────────────
    GpuTimer timer_sync;
    timer_sync.start();
    for (int i = 0; i < num_iters; ++i) {
        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), q_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), q_bytes, cudaMemcpyHostToDevice));
        attn.forward_sync(d_Q, d_K, d_V, d_Out, d_workspace);
        CHECK_CUDA(cudaMemcpy(h_Out_sync.data(), d_Out, q_bytes, cudaMemcpyDeviceToHost));
    }
    timer_sync.stop();
    float avg_sync_ms = timer_sync.elapsed_ms() / num_iters;

    // ── Benchmark: Streamed (2 streams) ─────────────────────────────────────
    GpuTimer timer_s2;
    timer_s2.start();
    for (int i = 0; i < num_iters; ++i) {
        attn.forward_streamed(h_Q.data(), h_K.data(), h_V.data(), h_Out_s2.data(), 2);
    }
    timer_s2.stop();
    float avg_s2_ms = timer_s2.elapsed_ms() / num_iters;

    // ── Benchmark: Streamed (4 streams) ─────────────────────────────────────
    GpuTimer timer_s4;
    timer_s4.start();
    for (int i = 0; i < num_iters; ++i) {
        attn.forward_streamed(h_Q.data(), h_K.data(), h_V.data(), h_Out_s4.data(), 4);
    }
    timer_s4.stop();
    float avg_s4_ms = timer_s4.elapsed_ms() / num_iters;

    BenchMetrics m{total_flops, dram_bytes, dma_bytes, avg_sync_ms, avg_s2_ms, avg_s4_ms};

    std::cout << ">>> Config: " << label << " (B=" << cfg.batch_size << ", H=" << cfg.num_heads
              << ", N=" << cfg.seq_len << ", D=" << cfg.head_dim << ")\n";
    std::cout << "    Total FLOPs             : " << std::scientific << std::setprecision(2) << m.flops
              << std::fixed << " FLOPs\n";
    std::cout << "    Kernel DRAM Traffic     : " << m.dram_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "    Arithmetic Intensity    : " << std::setprecision(2) << m.arithmetic_intensity() << " FLOP/Byte\n";
    std::cout << "    ----------------------------------------------------------------\n";
    std::cout << "    Sync Baseline Latency   : " << std::setprecision(3) << m.time_sync_ms << " ms  ("
              << std::setprecision(2) << m.tflops_sync() << " TFLOPS, "
              << m.eff_bw_GBs_sync() << " GB/s)\n";
    std::cout << "    Streamed (2 streams)    : " << std::setprecision(3) << m.time_stream2_ms << " ms\n";
    std::cout << "    Streamed (4 streams)    : " << std::setprecision(3) << m.time_stream4_ms << " ms  ("
              << std::setprecision(2) << m.tflops_stream4() << " TFLOPS, "
              << m.eff_bw_GBs_stream4() << " GB/s)\n";
    std::cout << "    >>> Stream Speedup      : " << std::setprecision(2) << m.speedup() << "x\n\n";

    return m;
}

int main() {
    print_device_info();

    std::cout << "========================================================================\n";
    std::cout << "                 END-TO-END ATTENTION BENCHMARK SUITE                   \n";
    std::cout << "========================================================================\n\n";

    // Workload 1: BERT-style medium sequence
    run_attention_benchmark("BERT-Base Workload", AttentionConfig{1, 12, 512, 64, true});

    // Workload 2: Multi-Batch Multi-Head
    run_attention_benchmark("Batched Multi-Head", AttentionConfig{4, 8, 256, 64, true});

    // Workload 3: Long Context sequence
    run_attention_benchmark("Long-Context Head", AttentionConfig{1, 4, 1024, 64, true});

    std::cout << "========================================================================\n";
    std::cout << "✅ BENCHMARK RUN COMPLETED!\n";
    std::cout << "========================================================================\n";
    return 0;
}

