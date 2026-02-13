#!/bin/bash
# Submit chained training jobs for qwen3_8b_base (10 epochs)

set -eo pipefail

ACCOUNT="coreai_horizon_dilations"
NUM_RUNS=3

JOB1=$(sbatch --account=$ACCOUNT --parsable experiments/eagle3/qwen3_8b_base/train.sbatch)
echo "Submitted job 1: $JOB1"

PREV_JOB=$JOB1
for i in $(seq 2 $NUM_RUNS); do
    NEXT_JOB=$(sbatch --account=$ACCOUNT --dependency=afterok:$PREV_JOB --parsable experiments/eagle3/qwen3_8b_base/train.sbatch)
    echo "Submitted job $i: $NEXT_JOB (depends on $PREV_JOB)"
    PREV_JOB=$NEXT_JOB
done

echo ""
echo "Submitted $NUM_RUNS chained jobs for qwen3_8b_base training"
echo "Monitor with: squeue -u \$USER"
