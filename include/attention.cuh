#pragma once

#include "common.cuh"
#include "matmul.cuh"
#include "mask.cuh"
#include "softmax.cuh"

// Config for attention computation
struct AttentionConfig {
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = 64;
    int head_dim = 64;
    bool causal = true;
};

// Midterm Concepts:
// - Pipelined execution across batches/heads using CUDA streams
// - cudaMallocHost / cudaFreeHost for pinned memory and DMA
// - cudaMemcpyAsync to overlap H2D, Kernel, and D2H
// - cudaEventRecord and cudaStreamWaitEvent for dependency coordination
// - Optional: CUDA Graphs capture for static launch graph execution

class NanoAttention {
public:
    AttentionConfig config;

    NanoAttention(AttentionConfig cfg) : config(cfg) {}

    // Synchronous baseline execution
    void forward_sync(const float* d_Q, const float* d_K, const float* d_V,
                      float* d_Out, float* d_workspace);

    // Asynchronous / pipelined execution with streams
    void forward_streamed(const float* h_Q, const float* h_K, const float* h_V,
                         float* h_Out, int num_streams);
};
