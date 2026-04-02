#!/bin/bash

N_RUNS=5

echo "============================================================"
echo "Submitting full P2P benchmark pipeline"
echo "============================================================"

# 1. Build
BUILD_JOB=$(sbatch --parsable build.sh)
echo "  Build job    : $BUILD_JOB"

# 2. N_RUNS run jobs — each depends on build succeeding
RUN_DEP="afterok:$BUILD_JOB"
RUN_JOBS=""
for i in $(seq 1 $N_RUNS); do
    JID=$(sbatch --parsable --dependency="$RUN_DEP" run.sh)
    echo "  Run job $i    : $JID"
    RUN_JOBS="${RUN_JOBS}:${JID}"
done

# 3. Plot job — depends on ALL run jobs succeeding
PLOT_JOB=$(sbatch --parsable --dependency="afterok${RUN_JOBS}" plot.sh)
echo "  Plot job     : $PLOT_JOB"

echo ""
echo "Pipeline submitted. Track with:"
echo "  squeue -u \$USER"
echo "============================================================"
