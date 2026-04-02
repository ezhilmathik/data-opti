# Paths derived from loaded modules  no hardcoding
LLVM_LIB  := $(shell clang++ --print-file-name=libomptarget.so | xargs dirname 2>/dev/null)
CUDA_PATH := $(CUDA_HOME)
GCC_LIB   := $(shell gcc -print-file-name=libstdc++.so | xargs dirname 2>/dev/null)

CLANG_FLAGS := \
    -fopenmp \
    -fopenmp-targets=nvptx64 \
    -Xopenmp-target=nvptx64 -march=sm_90 \
    --cuda-path=$(CUDA_PATH) \
    -Wl,-rpath,$(LLVM_LIB) \
    -O3

# nvcc: system CUDA
NVCC_FLAGS := -O3 -Xcompiler -fopenmp -arch=sm_90 -Xlinker -rpath,$(GCC_LIB)

# nvc++: NVHPC bundled CUDA
NVC_FLAGS  := -fast -mp=gpu -cuda -gpu=cc90 -Minfo=accel -lcudart

OMP_BINS  := version-1-off version-2-off version-4-off version-4-4-off
CUDA_BINS := version-1-cuda version-2-cuda version-4-cuda version-4-4-cuda
ACC_BINS  := version-1-acc version-2-acc version-4-acc version-4-4-acc
ALL_BINS  := $(OMP_BINS) $(CUDA_BINS) $(ACC_BINS)

.PHONY: all omp cuda acc clean

all: omp cuda acc

omp: $(OMP_BINS)

version-1-off: openmp-1.cc
	clang++ $(CLANG_FLAGS) -o $@ $<

version-2-off: openmp-2.cc
	clang++ $(CLANG_FLAGS) -o $@ $<

version-4-off: openmp-4.cc
	clang++ $(CLANG_FLAGS) -o $@ $<

version-4-4-off: openmp-4-4.cc
	clang++ $(CLANG_FLAGS) -o $@ $<

cuda: $(CUDA_BINS)

version-1-cuda: version-1.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

version-2-cuda: version-2.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

version-4-cuda: version-4.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

version-4-4-cuda: version-4-4.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

acc: $(ACC_BINS)

version-1-acc: version-1.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

version-2-acc: version-2.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

version-4-acc: version-4.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

version-4-4-acc: version-4-4.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

clean:
	rm -f $(ALL_BINS)
