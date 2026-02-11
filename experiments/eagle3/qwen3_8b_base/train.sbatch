#!/bin/bash
#SBATCH --job-name=eagle3-train-qwen3-8b-base
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

MODEL="Qwen/Qwen3-8B-Base"
MODEL_DIR="qwen3_8b_base"

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
HIDDEN_STATES_DIR="$LUSTRE_BASE/eagle3/hidden_states/$MODEL_DIR"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/ckpts/$MODEL_DIR"

CONTAINER="nvcr.io#nvidia/pytorch:25.01-py3"
MOUNTS="$LUSTRE_BASE:$LUSTRE_BASE"

srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$REPO_DIR" \
     bash -c "
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface

pip install --quiet --no-deps -e .
pip install --quiet accelerate datasets 'transformers>=4.51' 'huggingface_hub>=0.24.0'

cd examples/speculative_decoding

bash launch_train.sh \
    --model $MODEL \
    --mode eagle3 \
    --eagle_decoder_type llama \
    --offline-data $HIDDEN_STATES_DIR \
    --output_dir $OUTPUT_DIR \
    --num_epochs 2 \
    --lr 1e-4 \
    --train_bs 4 \
    --training_seq_len 2048 \
    --disable_tqdm True

echo 'Training complete for $MODEL'
echo 'Checkpoints: $OUTPUT_DIR'
"
