#!/bin/bash -l                            
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time 00:30:00                                              
#SBATCH --exclusive 
#SBATCH --account=p201103
#SBATCH --output=openmp-%j.out

#module load env/release/2023.1
#module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 # added 17/05/2024
#export NVCC_APPEND_FLAGS='-allow-unsupported-compiler' # added 10/04/2024

module purge
module load env/release/2025.1
module load LLVM/20.1.7-GCCcore-14.2.0
module load CUDA/12.8.0

clang++ -fopenmp \
        -fopenmp-targets=nvptx64-nvidia-cuda \
        -Xopenmp-target -march=sm_80 \
        --cuda-path=/apps/USE/easybuild/release/2025.1/software/CUDA/12.8.0 \
        -O3 -o version-2-off openmp-2.cc

clang++ -fopenmp \
        -fopenmp-targets=nvptx64-nvidia-cuda \
        -Xopenmp-target -march=sm_80 \
        --cuda-path=/apps/USE/easybuild/release/2025.1/software/CUDA/12.8.0 \
        -O3 -o version-4-off openmp-4.cc


Ns=(940000000, 1740000000, 2550000000, 3350000000, 4160000000, 4960000000)

for N in "${Ns[@]}"
do
    echo "Begin 2 GPU test"
    echo "Running OpenMP Off Together test 2 for N = $N"
    ./version-2-off "$N"
    sleep 15s
    echo "Begin 2 GPU test"
    echo "Running OpenMP Off Together test 4 for N = $N"
    ./version-4-off "$N"
    sleep 15s
    echo "==========================="
done
