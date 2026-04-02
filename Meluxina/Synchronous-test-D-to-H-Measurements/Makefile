LLVM_LIB  := /apps/USE/easybuild/release/2025.1/software/LLVM/20.1.7-GCCcore-14.2.0/lib
CUDA_PATH := /apps/USE/easybuild/release/2025.1/software/CUDA/12.8.0

CLANG_FLAGS := \
    -fopenmp \
    -fopenmp-targets=nvptx64-nvidia-cuda \
    -Xopenmp-target -march=sm_80 \
    --cuda-path=$(CUDA_PATH) \
    -Wl,-rpath,$(LLVM_LIB) \
    -O3

NVCC_FLAGS := -O3 -Xcompiler -fopenmp -arch=sm_80
NVC_FLAGS  := -fast -mp=gpu -cuda -gpu=cc80 -Minfo=accel -lcudart

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

cuda: $(CUDA_BINS)

version-1-cuda: version-1.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

version-2-cuda: version-2.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

version-4-cuda: version-4.cu
	nvcc $(NVCC_FLAGS) -o $@ $<

acc: $(ACC_BINS)

version-1-acc: version-1.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

version-2-acc: version-2.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

version-4-acc: version-4.cc
	nvc++ $(NVC_FLAGS) -o $@ $<

clean:
	rm -f $(ALL_BINS)
