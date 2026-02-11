#!/bin/bash
#SBATCH --job-name=eagle3-hs-qwen3-30b
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

MODEL="Qwen/Qwen3-30B-A3B"
DP_SIZE=4  # MoE 30B needs ~2 GPUs per instance, so DP=4

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
INPUT_DATA="$LUSTRE_BASE/eagle3/data/dapo.jsonl"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/hidden_states/qwen3_30b"

CONTAINER="nvcr.io#nvidia/pytorch:25.01-py3"
MOUNTS="$LUSTRE_BASE:$LUSTRE_BASE"

srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$REPO_DIR" \
     bash -c "
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface

pip install --quiet -e '.[hf]'
pip install --quiet accelerate transformers datasets

mkdir -p $OUTPUT_DIR

# Split input file for data-parallel processing
split -n l/$DP_SIZE --numeric-suffixes=0 -d --additional-suffix=.jsonl $INPUT_DATA /tmp/eagle3-hs-part-

# Each worker gets 2 GPUs via device_map=auto
for i in \$(seq 0 \$(($DP_SIZE - 1))); do
    GPU_START=\$((i * 2))
    GPU_END=\$((GPU_START + 1))
    PART_FILE=\$(printf '/tmp/eagle3-hs-part-%02d.jsonl' \"\$i\")
    CUDA_VISIBLE_DEVICES=\$GPU_START,\$GPU_END python examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py \
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