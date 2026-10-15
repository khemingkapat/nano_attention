# ROADMAP: NANO_ATTENTION (MIDTERM PREPARATION)

**Target Completion Date**: Tuesday, October 20, 2026 (3 days before exam on Friday, Oct 23, 2026)  
**Timeline**: 7 Days (Oct 14 – Oct 20)  
**Core Goal**: Implement a pure CUDA attention block from scratch to master the 5 core midterm topics:
1. Thread Hierarchy & Warp Divergence (`cuda_structure/`)
2. 2D Mapping & Memory Coalescing (`matrix/`)
3. Shared Memory Tiling & Dual Barriers (`gpu_memory/`)
4. Sequential Tree Reduction (`reduction/`)
5. Pinned Memory, Streams Pipelining & Events (`stream/`)

---

## 7-Day Day-by-Day Execution Plan

```mermaid
flowchart TD
    Day1["Day 1 (Oct 14): Project Setup, QK^T Naive & CPU Reference"] --> Day2["Day 2 (Oct 15): Tiled GEMM & Shared Memory Barriers"]
    Day2 --> Day3["Day 3 (Oct 16): Causal Masking, Scaling & Warp Divergence"]
    Day3 --> Day4["Day 4 (Oct 17): Numerically Stable Row-wise Softmax Reduction"]
    Day4 --> Day5["Day 5 (Oct 18): End-to-End Attention Integration (Sync Pipeline)"]
    Day5 --> Day6["Day 6 (Oct 19): CUDA Streams Pipelining & Event Overlap"]
    Day6 --> Day7["Day 7 (Oct 20): Multi-GPU Benchmarking (T4 vs RTX 4090) & Polish"]
```

---

### Day 1: Wednesday, Oct 14 – Baseline & Infrastructure
- [x] **Midterm Topic**: CUDA indexing basics & memory management (`cuda_structure/`).
- [x] **Milestone**:
  - [x] Implement basic data allocation (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
  - [x] Wire up CPU reference test harness in `tests/test_matmul.cu` (was test_day1.cu) with synthetic data.
  - [x] Implement naive GEMM ($Q \times K^T$) as a baseline to prove end-to-end data transfer works.
- [x] **Self-Check / Exam Question Check**:
  - [x] *Can you write the 1D and 2D global index equations from memory?*
  - [x] *Are bounds checks (`if (row < M && col < N)`) strictly enforced?*

---

### Day 2: Thursday, Oct 15 – Tiled Matrix Multiplication (`gpu_memory/`)
- [x] **Midterm Topic**: Shared memory tiling, bank conflicts, dual barriers, arithmetic intensity.
- [x] **Milestone**:
  - [x] Implement `matmul_tiled_kernel` in `src/matmul.cu` with `__shared__ float s_A[TILE][TILE]` and `s_B[TILE][TILE]`.
  - [x] Handle both non-transposed ($S \times V$) and transposed ($Q \times K^T$) layouts.
  - [x] Enforce boundary zero-padding for non-multiple-of-tile dimensions.
  - [x] Verify against CPU GEMM.
- [x] **Self-Check / Exam Question Check**:
  - [x] *Why are two `__syncthreads()` mandatory per tile iteration?*
  - [x] *What is the arithmetic intensity improvement over naive GEMM?*

---

### Day 3: Friday, Oct 16 – Scaling, 2D Coalescing & Masking (`matrix/`, `cuda_structure/`)
- [ ] **Midterm Topic**: Global memory coalescing (128B bursts) & warp divergence avoidance.
- [ ] **Milestone**:
  - Implement `scale_and_mask_kernel` in `src/mask.cu`.
  - Scale scores by $1/\sqrt{d_k}$.
  - Apply causal lower-triangular mask ($j > i \implies -\infty$).
  - Experiment with thread block shapes: compare `(32, 8)` vs `(8, 32)` to observe coalesced access patterns.
  - Test branchless masking to eliminate intra-warp divergence.
- [ ] **Self-Check / Exam Question Check**:
  - *Why does block `(32, 8)` outperform `(8, 32)` for row-major access?*
  - *How does intra-warp branching (`tid % 2 == 0`) compare with warp-aligned branching?*

---

### Day 4: Saturday, Oct 17 – Row-Wise Softmax with Tree Reduction (`reduction/`)
- [ ] **Midterm Topic**: Tree reduction, sequential addressing, thread coarsening, numerical precision.
- [ ] **Milestone**:
  - Implement `softmax_forward_kernel` in `src/softmax.cu`.
  - Step 1: Find row maximum $m = \max_j(x_{ij})$ via sequential tree reduction (`tid < s`).
  - Step 2: Compute $\sum_j \exp(x_{ij} - m)$ via tree reduction.
  - Step 3: Normalize elements: $\exp(x_{ij} - m) / \sum$.
  - Verify floating-point stability (no NaN/inf) against CPU reference.
- [ ] **Self-Check / Exam Question Check**:
  - *Why does sequential addressing eliminate warp divergence until `s < 32`?*
  - *Why subtract the row maximum before computing `exp()`?*

---

### Day 5: Sunday, Oct 18 – End-to-End Synchronous Attention
- [ ] **Midterm Topic**: Composing kernels, device synchronization, memory workspace management.
- [ ] **Milestone**:
  - Assemble the full pipeline in `src/attention.cu`:
    $$\text{Q} \xrightarrow{\text{Tiled GEMM}} QK^T \xrightarrow{\text{Scale+Mask}} S \xrightarrow{\text{Softmax}} P \xrightarrow{\text{Tiled GEMM}} PV = \text{Out}$$
  - Run full test suite across varied batch sizes ($B=1, 2, 4$) and sequence lengths ($N=32, 64, 128$).
  - Ensure zero memory leaks with `cudaMemGetInfo` or `cudaDeviceReset`.
- [ ] **Self-Check / Exam Question Check**:
  - *Can you trace the memory footprint of temporary activation buffers?*

---

### Day 6: Monday, Oct 19 – Streams, Pinned Memory & Pipelining (`stream/`)
- [ ] **Midterm Topic**: Asynchronous transfers, pinned memory (`cudaMallocHost`), multi-stream concurrency, event timing.
- [ ] **Milestone**:
  - Implement `forward_streamed` in `src/attention.cu`.
  - Allocate pinned host memory for $Q, K, V, \text{Out}$.
  - Create multiple CUDA streams (`cudaStreamCreate`) to process heads or batches concurrently.
  - Pipeline chunks: Overlap `H2D(chunk i+1)` with `Compute(chunk i)` and `D2H(chunk i-1)`.
  - Use `cudaEventRecord` and `cudaStreamWaitEvent` for timing and dependencies.
- [ ] **Self-Check / Exam Question Check**:
  - *Why does `cudaMemcpyAsync` require pinned (page-locked) memory?*
  - *How do CUDA events measure elapsed time without CPU synchronization stalls?*

---

### Day 7: Tuesday, Oct 20 – Benchmarking & Midterm Readiness Wrap-Up
- [ ] **Midterm Topic**: Hardware performance profiling (T4 sm_75 vs RTX 4090 sm_89), theoretical vs effective bandwidth.
- [ ] **Milestone**:
  - Run benchmark harness in `benchmarks/bench_attention.cu`.
  - Measure kernel latency (ms) with `GpuTimer` and calculate effective bandwidth (GB/s).
  - Deploy to Google Colab T4 via `colab-submit` and record metrics.
  - Deploy to KMUTT Slurm cluster (RTX 4090) via `slurm-scp` / `sbatch` and record metrics.
  - Verify stream speedup vs synchronous baseline.
- [ ] **Outcome**: Clean, documented repository ready as your ultimate midterm reference!

---

## Buffer / Review Period (Oct 21 – Oct 22)
- **Wednesday, Oct 21**: Review past exam papers and theoretical calculations (Peak BW equation, roofline model).
- **Thursday, Oct 22**: Final mental walkthrough of key kernel code snippets and indexing formulas.
- **Friday, Oct 23 (Exam Day)**: Ace the exam!
