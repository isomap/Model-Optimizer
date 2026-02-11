#!/bin/bash
#SBATCH --job-name=eagle3-hs-gpt-oss-120b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=batch_long
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

MODEL="openai/gpt-oss-120b"
# 120B needs all 8 GPUs for one instance, no data parallelism

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
INPUT_DATA="$LUSTRE_BASE/eagle3/data/dapo.jsonl"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/hidden_states/gpt_oss_120b"

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

# Single process using all 8 GPUs via device_map=auto
python examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py \
    --model $MODEL \
    --input-data $INPUT_DATA \
    --output-dir $OUTPUT_DIR \
    --max-seq-len 3072

echo 'Hidden state extraction complete for $MODEL'
echo 'Output files:'
ls $OUTPUT_DIR | wc -l
"