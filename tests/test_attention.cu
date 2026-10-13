#include "common.cuh"
#include "attention.cuh"
#include <cmath>
#include <iostream>
#include <vector>

// Reference CPU implementation for Attention(Q, K, V)
void cpu_attention_ref(const float* Q, const float* K, const float* V, float* Out,
                       int seq_len, int d_k, bool causal) {
    float scale = 1.0f / std::sqrt((float)d_k);
    std::vector<float> scores(seq_len * seq_len);

    // 1. Q @ K^T & scale & causal mask
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (causal && j > i) {
                scores[i * seq_len + j] = -1e9f;
            } else {
                float sum = 0.0f;
                for (int k = 0; k < d_k; ++k) {
                    sum += Q[i * d_k + k] * K[j * d_k + k];
                }
                scores[i * seq_len + j] = sum * scale;
            }
        }
    }

    // 2. Softmax along each row
    for (int i = 0; i < seq_len; ++i) {
        float max_val = -1e9f;
        for (int j = 0; j < seq_len; ++j) {
            if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_val);
            sum_exp += scores[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] /= sum_exp;
        }
    }

    // 3. Scores @ V
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_k; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; ++k) {
                sum += scores[i * seq_len + k] * V[k * d_k + j];
            }
            Out[i * d_k + j] = sum;
        }
    }
}

int main() {
    std::cout << "=== Correctness Test: NanoAttention ===" << std::endl;
    // Scaffold test runner
    std::cout << "[INFO] CPU reference generator ready. Implement kernels to run full diff test!" << std::endl;
    return 0;
}
