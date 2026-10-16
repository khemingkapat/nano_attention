#include "attention.cuh"
#include <cmath>

void NanoAttention::forward_sync(const float *d_Q, const float *d_K, const float *d_V, float *d_Out,
                                 float *d_workspace) {
    // Pipeline:
    // 1. Q @ K^T -> workspace (size: seq_len x seq_len per head/batch)
    // 2. Scale by 1.0 / sqrt(head_dim) and apply causal mask
    // 3. Softmax along each row of workspace
    // 4. workspace @ V -> d_Out

    float scale = 1.0f / sqrtf((float)config.head_dim);
    int total_matrices = config.batch_size * config.num_heads;

    for (int i = 0; i < total_matrices; ++i) {
        int qk_offset = i * config.seq_len * config.head_dim;
        int score_offset = i * config.seq_len * config.seq_len;
        int out_offset = i * config.seq_len * config.head_dim;

        // Step 1: Q @ K^T
        launch_matmul(d_Q + qk_offset, d_K + qk_offset, d_workspace + score_offset, config.seq_len,
                      config.head_dim, config.seq_len, true, 0);

        // Step 2: Scale & Mask
        launch_scale_and_mask(d_workspace + score_offset, 1, config.seq_len, scale, config.causal,
                              0);

        // Step 3: Softmax
        launch_softmax(d_workspace + score_offset, 1, config.seq_len, 0);

        // Step 4: Scores @ V
        launch_matmul(d_workspace + score_offset, d_V + qk_offset, d_Out + out_offset,
                      config.seq_len, config.seq_len, config.head_dim, false, 0);
    }
}

void NanoAttention::forward_streamed(const float *h_Q, const float *h_K, const float *h_V,
                                     float *h_Out, int num_streams) {
    // TODO: Implement stream pipelining across heads/batches!
    // Concepts:
    // - cudaStreamCreate
    // - cudaMemcpyAsync (H2D chunk i+1, Kernel chunk i, D2H chunk i-1)
    // - cudaStreamSynchronize / cudaEventRecord
}
