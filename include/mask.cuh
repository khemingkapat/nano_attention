#pragma once
// PUBLIC INTERFACE ONLY. Kernel lives in src/mask.cu.

#include "common.cuh"

// In-place on scores[batch_size * seq_len, seq_len]:
//   scores *= scale; if causal, scores[i][j] = -inf-ish for j > i
void launch_scale_and_mask(float *d_scores, int batch_size, int seq_len, float scale,
                           bool causal = true, cudaStream_t stream = 0);
