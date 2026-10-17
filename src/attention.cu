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
    cudaStream_t streams[num_streams];

    size_t qk_bytes = (size_t)num_streams * config.seq_len * config.head_dim * sizeof(float);
    size_t ws_bytes = (size_t)num_streams * config.seq_len * config.seq_len * sizeof(float);

    int total_matrices = config.batch_size * config.num_heads;

    float *d_Q, *d_K, *d_V, *d_Out, *d_workspace;
    cudaMalloc(&d_Q, qk_bytes);
    cudaMalloc(&d_K, qk_bytes);
    cudaMalloc(&d_V, qk_bytes);
    cudaMalloc(&d_Out, qk_bytes);
    cudaMalloc(&d_workspace, ws_bytes);

    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s]);
    }

    float scale = 1.0f / sqrtf((float)config.head_dim);
    for (int i = 0; i < total_matrices; i++) {
        int s = i % num_streams;
        int d_offset = s * config.seq_len * config.head_dim; // offset by streams
                                                             //
        size_t chunk_qk_bytes = (size_t)config.seq_len * config.head_dim * sizeof(float);
        int h_offset = i * config.seq_len * config.head_dim;
        int ws_offset = s * config.seq_len * config.seq_len;

        cudaMemcpyAsync(d_Q + d_offset, h_Q + h_offset, chunk_qk_bytes, cudaMemcpyHostToDevice,
                        streams[s]);
        cudaMemcpyAsync(d_K + d_offset, h_K + h_offset, chunk_qk_bytes, cudaMemcpyHostToDevice,
                        streams[s]);
        cudaMemcpyAsync(d_V + d_offset, h_V + h_offset, chunk_qk_bytes, cudaMemcpyHostToDevice,
                        streams[s]);

        launch_matmul(d_Q + d_offset, d_K + d_offset, d_workspace + ws_offset, config.seq_len,
                      config.head_dim, config.seq_len, true, streams[s]);
        launch_scale_and_mask(d_workspace + ws_offset, 1, config.seq_len, scale, config.causal,
                              streams[s]);
        launch_softmax(d_workspace + ws_offset, 1, config.seq_len, streams[s]);
        launch_matmul(d_workspace + ws_offset, d_V + d_offset, d_Out + d_offset, config.seq_len,
                      config.seq_len, config.head_dim, false, streams[s]);

        cudaMemcpyAsync(h_Out + h_offset, d_Out + d_offset, chunk_qk_bytes, cudaMemcpyDeviceToHost,
                        streams[s]);
    }
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
    }

    for (int s = 0; s < num_streams; s++) {
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Out);
    cudaFree(d_workspace);
}
