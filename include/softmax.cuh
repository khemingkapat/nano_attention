#pragma once
// PUBLIC INTERFACE ONLY. Kernel lives in src/softmax.cu.

#include "common.cuh"

// In-place row-wise softmax on scores[batch_size * seq_len, seq_len]
void launch_softmax(float *d_scores, int batch_size, int seq_len, cudaStream_t stream = 0);
