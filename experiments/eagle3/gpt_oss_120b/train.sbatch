#!/bin/bash
#SBATCH --job-name=eagle3-train-gpt-oss-120b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=experiments/eagle3/gpt_oss_120b/logs/%j.out
#SBATCH --error=experiments/eagle3/gpt_oss_120b/logs/%j.err

set -eo pipefail

MODEL="openai/gpt-oss-120b"

LUSTRE_BASE="/lustre/fsw/portfolios/general/users/$USER"
REPO_DIR="$LUSTRE_BASE/repos/Model-Optimizer"
HIDDEN_STATES_DIR="$LUSTRE_BASE/eagle3/hidden_states/gpt_oss_120b"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/ckpts/gpt_oss_120b"
HF_HOME="$LUSTRE_BASE/.cache/huggingface"
export HF_HOME

cd "$REPO_DIR" && git pull
pip install -e ".[hf]"
pip install accelerate transformers datasets

cd examples/speculative_decoding

bash launch_train.sh \
    --model "$MODEL" \
    --mode eagle3 \
    --eagle_decoder_type llama \
    --offline-data "$HIDDEN_STATES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 2 \
    --lr 1e-4 \
    --train_bs 4 \
    --training_seq_len 2048 \
    --disable_tqdm True

echo "Training complete for $MODEL"
echo "Checkpoints saved to $OUTPUT_DIR"