#!/bin/bash
#SBATCH --job-name=eagle3-hs-gpt-oss-120b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=backfill
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

MODEL="openai/gpt-oss-120b"

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
GEN_DIR="$LUSTRE_BASE/eagle3/generated/gpt_oss_120b"
RAW_DATA="$GEN_DIR/conversations_combined.jsonl"
INPUT_DATA="$GEN_DIR/conversations_clean.jsonl"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/hidden_states/gpt_oss_120b"

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

# Combine shard outputs into single file
echo 'Combining shard outputs...'
cat $GEN_DIR/output-*.jsonl > $RAW_DATA
echo "Combined \$(wc -l < $RAW_DATA) lines"

# Validate and clean generated conversations
python3 $REPO_DIR/experiments/eagle3/data/validate_generated.py \
    --input $RAW_DATA \
    --output $INPUT_DATA

mkdir -p $OUTPUT_DIR

python examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py \
    --model $MODEL \
    --input-data $INPUT_DATA \
    --output-dir $OUTPUT_DIR \
    --max-seq-len 32768

echo 'Hidden state extraction complete for $MODEL'
ls $OUTPUT_DIR | wc -l
"
