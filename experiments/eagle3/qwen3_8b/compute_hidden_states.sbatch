#!/bin/bash
#SBATCH --job-name=eagle3-hs-qwen3-8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

MODEL="Qwen/Qwen3-8B"
DP_SIZE=8  # 8B model fits on 1 GPU, so DP=8

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
INPUT_DATA="$LUSTRE_BASE/eagle3/data/dapo.jsonl"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/hidden_states/qwen3_8b"

CONTAINER="nvcr.io#nvidia/pytorch:25.01-py3"
MOUNTS="$LUSTRE_BASE:$LUSTRE_BASE"

srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$REPO_DIR" \
     bash -c "
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface

# Only install deps needed for hidden state extraction (not modelopt, which upgrades torch and breaks torchvision)
pip install --quiet 'transformers>=4.51' accelerate datasets

mkdir -p $OUTPUT_DIR

# Split input file for data-parallel processing
split -n l/$DP_SIZE --numeric-suffixes=0 -d --additional-suffix=.jsonl $INPUT_DATA /tmp/eagle3-hs-part-

for i in \$(seq 0 \$(($DP_SIZE - 1))); do
    PART_FILE=\$(printf '/tmp/eagle3-hs-part-%02d.jsonl' \"\$i\")
    CUDA_VISIBLE_DEVICES=\$i python examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py \
        --model $MODEL \
        --input-data \"\$PART_FILE\" \
        --output-dir $OUTPUT_DIR \
        --max-seq-len 3072 &
done
wait

rm -f /tmp/eagle3-hs-part-*.jsonl

echo 'Hidden state extraction complete for $MODEL'
echo 'Output files:'
ls $OUTPUT_DIR | wc -l
"