NVCC := nvcc
CXXFLAGS := -O3 -std=c++17 -Iinclude

# ── Target GPU ──────────────────────────────────────────────────────────────
#   make push-matmul              -> A100 (sm_80, default)
#   make push-matmul GPU=T4       -> sm_75
#   make test-matmul GPU=4090     -> sm_89 (Slurm)
#   ARCH=all                      -> fat binary for every GPU below
GPU ?= A100
ARCH_A100 := 80
ARCH_T4   := 75
ARCH_L4   := 89
ARCH_4090 := 89
ARCH_H100 := 90
ARCH ?= $(ARCH_$(GPU))
ifeq ($(ARCH),)
    $(error Unknown GPU=$(GPU). Use A100, T4, L4, 4090, H100 or set ARCH=<cc> directly)
endif

ARCH_FLAGS := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_89,code=sm_89
ifeq ($(ARCH), all)
    ARCH_TARGET := $(ARCH_FLAGS)
else
    ARCH_TARGET := -arch=sm_$(ARCH)
endif

SRCS := src/matmul.cu src/softmax.cu src/mask.cu src/attention.cu
OBJS := $(SRCS:.cu=.o)

BIN_DIR := bin
$(shell mkdir -p $(BIN_DIR))

all: $(BIN_DIR)/main $(BIN_DIR)/test_attention $(BIN_DIR)/bench_attention

$(BIN_DIR)/main: src/main.cu $(OBJS)
	$(NVCC) $(CXXFLAGS) $(ARCH_TARGET) $^ -o $@

$(BIN_DIR)/test_attention: tests/test_attention.cu $(OBJS)
	$(NVCC) $(CXXFLAGS) $(ARCH_TARGET) $^ -o $@

$(BIN_DIR)/bench_attention: benchmarks/bench_attention.cu $(OBJS)
	$(NVCC) $(CXXFLAGS) $(ARCH_TARGET) $^ -o $@

src/%.o: src/%.cu include/%.cuh include/common.cuh
	$(NVCC) $(CXXFLAGS) $(ARCH_TARGET) -c $< -o $@

format:
	clang-format -i include/*.cuh src/*.cu tests/*.cu benchmarks/*.cu

clean:
	rm -rf src/*.o $(BIN_DIR) build

test: $(BIN_DIR)/test_attention
	./$(BIN_DIR)/test_attention

bench: $(BIN_DIR)/bench_attention
	./$(BIN_DIR)/bench_attention

# ── Per-component tests (local GPU) ─────────────────────────────────────────
#   make test-matmul   -> builds tests/test_matmul.cu + all src objs, runs it
$(BIN_DIR)/test_%: tests/test_%.cu $(OBJS) include/test_utils.cuh
	$(NVCC) $(CXXFLAGS) $(ARCH_TARGET) tests/test_$*.cu $(OBJS) -o $@

test-%: $(BIN_DIR)/test_%
	./$<

# ── Single-file bundles for Colab ───────────────────────────────────────────
#   make bundle-matmul  -> build/test_matmul.cu (self-contained)
#   make push-matmul    -> bundle + colab-push to /content/test_matmul.cu
bundle-bench: FORCE
	python3 scripts/bundle.py benchmarks/bench_attention.cu build/bench_attention.cu

push-bench: bundle-bench
	colab-push build/bench_attention.cu
	@echo "On VM ($(GPU)):  nvcc bench_attention.cu -o bench -arch=sm_$(ARCH) -std=c++17 && ./bench"

bundle-%: FORCE
	python3 scripts/bundle.py tests/test_$*.cu build/test_$*.cu

push-%: bundle-%
	colab-push build/test_$*.cu
	@echo "On VM ($(GPU)):  nvcc test_$*.cu -o t -arch=sm_$(ARCH) -std=c++17 && ./t"

FORCE:
.SECONDARY:
.PHONY: all clean format test bench FORCE
