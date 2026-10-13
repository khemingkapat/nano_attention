#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                                          \
    do {                                                                                          \
        cudaError_t err_ = (call);                                                                \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(err_), cudaGetErrorName(err_));                            \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                                                   \
    do {                                                                                          \
        cudaError_t err_ = cudaGetLastError();                                                    \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s (%s)\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_), cudaGetErrorName(err_));                            \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(start_event, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(stop_event, stream));
    }

    float elapsed_ms() {
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};
