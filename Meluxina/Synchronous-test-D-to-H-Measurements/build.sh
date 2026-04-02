#!/bin/bash -l
#SBATCH --job-name=build
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=00:30:00
#SBATCH --account=p201103
#SBATCH --output=build-%j.out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — OpenMP offload (clang++) + CUDA (nvcc)
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 1: OpenMP offload + CUDA  (no NVHPC)"
echo "============================================================"

module purge
module load env/release/2025.1
module load LLVM/20.1.7-GCCcore-14.2.0
module load CUDA/12.8.0

make omp
make cuda

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — OpenACC (nvc++)
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 2: OpenACC  (NVHPC)"
echo "============================================================"

module purge
module load env/release/2025.1
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

make acc

# ─────────────────────────────────────────────────────────────────────────────
# Verify ALL binaries exist before declaring success
# If any are missing the job exits non-zero -> afterok dependency blocks run jobs
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Verifying binaries..."
echo "============================================================"

EXPECTED=(
    version-1-off   version-2-off   version-4-off
    version-1-cuda  version-2-cuda  version-4-cuda 
    version-1-acc   version-2-acc   version-4-acc
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
echo "All binaries present — run jobs are cleared to start."
echo "============================================================"
