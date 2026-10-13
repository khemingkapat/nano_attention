NVCC := nvcc
CXXFLAGS := -O3 -std=c++17 -Iinclude

# Architectures: sm_75 (Tesla T4 / Colab), sm_89 (RTX 4090 / Slurm)
ARCH_FLAGS := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89

# Local build defaults to native or sm_75 if no target specified
ARCH ?= 75
ifeq ($(ARCH), 75)
    ARCH_TARGET := -arch=sm_75
else ifeq ($(ARCH), 89)
    ARCH_TARGET := -arch=sm_89
else
    ARCH_TARGET := $(ARCH_FLAGS)
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
	rm -rf src/*.o $(BIN_DIR)

test: $(BIN_DIR)/test_attention
	./$(BIN_DIR)/test_attention

bench: $(BIN_DIR)/bench_attention
	./$(BIN_DIR)/bench_attention

.PHONY: all clean format test bench
