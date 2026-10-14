# nano_attention

A bare-metal, pure CUDA implementation of the **Attention Mechanism** built from scratch for CPE 393 GPU Programming midterm preparation (Oct 23, 2026).

Target Architectures:
- **NVIDIA A100** (`sm_80`, Colab — default)
- **NVIDIA Tesla T4** (`sm_75`, Colab — legacy)
- **NVIDIA RTX 4090** (`sm_89`, Slurm Cluster)

---

## Architecture & Midterm Concept Mapping

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

| Component | Midterm Concept | Key Technique to Implement |
| :--- | :--- | :--- |
| **`matmul.cuh`** | Shared Memory Tiling (`gpu_memory/`) | `__shared__` tiles, double `__syncthreads()`, zero-padding boundary guards for arbitrary sequence lengths. |
| **`mask.cuh`** | SIMT & 2D Indexing (`cuda_structure/`, `matrix/`) | Coalesced 128B DRAM bursts (row-major mapping), branchless causal masking to avoid warp divergence. |
| **`softmax.cuh`** | Parallel Tree Reduction (`reduction/`) | Sequential addressing (`tid < s`) tree reduction for $\max$ and $\sum \exp$, avoiding warp divergence. |
| **`attention.cuh`** | CUDA Streams & Events (`stream/`) | Overlapping H2D, Kernel, and D2H via `cudaMemcpyAsync`, pinned host memory (`cudaMallocHost`), event sync. |

---

## Directory Structure

```text
nano_attention/
├── flake.nix              # Reproducible Nix dev shell (CUDA, gcc, clangd, colab/slurm shorthands)
├── .envrc                 # direnv configuration
├── .clang-format          # LLVM/Google C++ formatting
├── Makefile               # Build configuration (GPU=A100|T4|4090, default A100)
├── include/
│   ├── common.cuh         # CHECK_CUDA, CHECK_LAST_CUDA_ERROR, GpuTimer
│   ├── test_utils.cuh     # Host-only: DeviceBuffer, random_vector, verify, CPU refs
│   ├── matmul.cuh         # Tiled matrix multiplication interface
│   ├── mask.cuh           # Scale & causal masking interface
│   ├── softmax.cuh        # Row-wise reduction softmax interface
│   └── attention.cuh      # NanoAttention orchestrator & stream pipeline
├── src/
│   ├── matmul.cu          # MatMul kernel launch stubs
│   ├── mask.cu            # Masking kernel launch stubs
│   ├── softmax.cu         # Softmax kernel launch stubs
│   ├── attention.cu       # Attention forward sync & stream stubs
│   └── main.cu            # CLI entry point
├── scripts/
│   └── bundle.py          # Flattens a test + deps into build/*.cu for colab-push
├── tests/
│   ├── test_matmul.cu     # make test-matmul / make push-matmul
│   └── test_attention.cu  # Verification against CPU reference
└── benchmarks/
    └── bench_attention.cu # Latency & effective bandwidth measurement
```

---

## Development Workflow

### 1. Nix & Direnv
If `direnv` is enabled:
```bash
direnv allow
```

### 2. Building
```bash
# Compile for sm_75 (Tesla T4) — default is GPU=A100 (sm_80)
make ARCH=75

# Compile for sm_89 (RTX 4090)
make ARCH=89

# Format code
make format
```

### 3. Testing & Benchmarking
```bash
make test
make bench
```

### 4. Remote Execution
- **Colab (A100)**:
  ```bash
  colab-start A100 cuda-dev
  make push-matmul   # then build in colab-console
  ```
- **Slurm (RTX 4090)**:
  ```bash
  slurm-scp . :nano_attention/
  slurm-ssh
  ```
