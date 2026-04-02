#!/bin/bash -l                                                                                   
#SBATCH --job-name=build                                                                       
#SBATCH --time=00:15:00                                                                          
#SBATCH --account=ehpc229                                                                        
#SBATCH --partition=acc                                                                          
#SBATCH --qos=acc_ehpc                                                                           
#SBATCH --nodes=1                                                                                
#SBATCH --exclusive                                                                               
#SBATCH --output=build-%j.out


# -------------------------------------------------------------------------------------------
# Stage 1  OpenMP offload (clang++) + CUDA (nvcc)
# -------------------------------------------------------------------------------------------
echo "============================================================"
echo "Stage 1: OpenMP offload + CUDA  (no NVHPC)"
echo "============================================================"

module purge
module load EB/apps
module load GCC/13.2.0
module load cuda/12.8
module load clang/18.1.8-cuda12.8


make omp
make cuda

# -------------------------------------------------------------------------------------------
# Stage 2  OpenACC (nvc++)
# -------------------------------------------------------------------------------------------
echo "============================================================"
echo "Stage 2: OpenACC  (NVHPC)"
echo "============================================================"

module purge
module load EB/apps
module load NVHPC/23.7-CUDA-12.2.0

make acc

# -------------------------------------------------------------------------------------------
# Verify ALL binaries exist before declaring success
# -------------------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Verifying binaries..."
echo "============================================================"

EXPECTED=(
    version-1-off   version-2-off   version-4-off   version-4-4-off
    version-1-cuda  version-2-cuda  version-4-cuda  version-4-4-cuda
    version-1-acc   version-2-acc   version-4-acc   version-4-4-acc
)

MISSING=()
for bin in "${EXPECTED[@]}"; do
    if [[ -x "./$bin" ]]; then
        echo "  OK : $bin"
    else
        echo "  MISSING : $bin"
        MISSING+=("$bin")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: ${#MISSING[@]} binary/binaries missing: ${MISSING[*]}"
    echo "Run jobs will NOT start (afterok dependency will block them)."
    exit 1
fi

echo ""
echo "All binaries present  run jobs are cleared to start."
echo "============================================================"
