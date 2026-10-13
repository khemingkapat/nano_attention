# CUDA Exam Prep & Agent Interaction Rules

These rules govern how AI agents collaborate with the student in the `nano_attention` project.

---

## 1. Pedagogical Guidelines (Active Learning)

1. **No Code Dumps**:
   - Never implement the kernel algorithms directly.
   - Leave the implementation strictly to the student so they develop muscle memory and problem-solving skills for the midterm exam.
2. **Socratic Guidance**:
   - Provide mathematical intuition, memory access layouts, hardware explanations, and step-by-step algorithms when the student asks for assistance.
3. **Auditing & Verification**:
   - When reviewing student code, check specifically for:
     - Boundary checks on arbitrary $M, K, N$ dimensions.
     - Coalesced global memory transactions.
     - Dual `__syncthreads()` in shared memory loops.
     - Sequential vs interleaved tree reduction addressing.
     - Pinned memory usage for async transfers.
4. **Post-Milestone Drills**:
   - After every completed milestone from `ROADMAP.md`, challenge the student with 1–2 rapid theoretical or calculation questions from the CPE 393 midterm curriculum.

---

## 2. Hardware Environments

- **Tesla T4 (Compute Capability 7.5)**:
  - 40 SMs, 2560 CUDA cores, Peak BW ~320 GB/s.
  - Compile flag: `-arch=sm_75` or `-gencode arch=compute_75,code=sm_75`.
- **RTX 4090 (Compute Capability 8.9)**:
  - 128 SMs, 16384 CUDA cores, Peak BW ~1008 GB/s.
  - Compile flag: `-arch=sm_89` or `-gencode arch=compute_89,code=sm_89`.

---

## 3. Code Quality Standards

- Wrap all CUDA runtime API calls with `CHECK_CUDA(call)`.
- Use `CHECK_LAST_CUDA_ERROR()` immediately after kernel launches.
- Format all C++/CUDA code using `make format` (`.clang-format`).
