#include "common.cuh"
#include "attention.cuh"
#include <iostream>

int main() {
    std::cout << "=== Benchmark: NanoAttention ===" << std::endl;
    std::cout << "Target GPUs: T4 (sm_75) and RTX 4090 (sm_89)" << std::endl;
    std::cout << "Metrics to measure: Latency (ms), Effective Bandwidth (GB/s), Streams speedup" << std::endl;
    return 0;
}
