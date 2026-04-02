#!/bin/bash
#SBATCH --job-name=cray-test
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --account=project_465002427
#SBATCH --time=00:30:00
#SBATCH --exclusive

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4

export HSA_ENABLE_SDMA=0
export OMP_TARGET_OFFLOAD=mandatory
#export LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES=0
OMPTARGET_P2P_LIB="/scratch/project_465002427/omptarget-p2p/lib"


# HIP builds
amdclang++ -O3 -fopenmp -x hip --offload-arch=gfx90a -o version-1-hip hip-1.hip -Wno-unused-result
#amdclang++ -O3 -fopenmp -x hip --offload-arch=gfx90a -o version-4-hip hip-4.hip -Wno-unused-result

# OpenMP offload builds
CC -fopenmp \
  -fopenmp-targets=amdgcn-amd-amdhsa \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
  -O3 -o version-1-off openmp-1-1.cc

#CC -fopenmp \
#  -fopenmp-targets=amdgcn-amd-amdhsa \
#  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
#  -O3 -o version-4-off openmp-4.cc

Ns=(117500000 217500000 318750000 418750000 520000000 1240000000)
Ns_4to4=(58750000 108750000 159375000 209375000 260000000 310000000)

for N in "${Ns[@]}"
do
    echo "Running OpenMP Off test for N = $N"
    LD_PRELOAD="$OMPTARGET_P2P_LIB/libomp.so:$OMPTARGET_P2P_LIB/libomptarget.so.19.0git" \
      ./version-1-off "$N"
    sleep 15s

    echo "Running HIP test for N = $N"
    ./version-1-hip "$N"
    sleep 15s
done
