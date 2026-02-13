#!/bin/bash
#SBATCH --job-name=eagle3-train-qwen3_32b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

MODEL="Qwen/Qwen3-32B"
MODEL_DIR="qwen3_32b"
NUM_EPOCHS=10

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
DATA_PATH="$LUSTRE_BASE/eagle3/generated/$MODEL_DIR/conversations_aggregated.jsonl"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/training/$MODEL_DIR"
CHECKPOINT_DIR="$OUTPUT_DIR/ckpts"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$CHECKPOINT_DIR"

CONTAINER="nvcr.io#nvidia/pytorch:25.01-py3"
MOUNTS="$LUSTRE_BASE:$LUSTRE_BASE"

srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$REPO_DIR" \
     bash -c "
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface
export TOKENIZERS_PARALLELISM=False

pip install --quiet --no-deps -e .
pip install --quiet accelerate datasets 'transformers>=4.51' 'huggingface_hub>=0.24.0'

cd examples/speculative_decoding

# Check for existing checkpoint to resume
RESUME_ARG=\"\"
if [ -d \"$CHECKPOINT_DIR\" ] && [ \"\$(ls -A $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | wc -l)\" -gt 0 ]; then
    LATEST_CKPT=\$(ls -td $CHECKPOINT_DIR/checkpoint-* | head -1)
    echo \"Resuming from: \$LATEST_CKPT\"
    RESUME_ARG=\"--resume_from_checkpoint \$LATEST_CKPT\"
fi

# Create FSDP config
cat > fsdp_config.json << 'FSDP_EOF'
{
  \"fsdp_transformer_layer_cls_to_wrap\": [\"LlamaDecoderLayer\", \"Qwen2DecoderLayer\"],
  \"fsdp_backward_prefetch\": \"backward_pre\",
  \"fsdp_state_dict_type\": \"full_state_dict\",
  \"fsdp_auto_wrap_policy\": \"TRANSFORMER_BASED_WRAP\",
  \"fsdp_use_orig_params\": true
}
FSDP_EOF

bash launch_train.sh \
    --model $MODEL \
    --data $DATA_PATH \
    --mode eagle3 \
    --eagle_decoder_type llama \
    --output_dir $CHECKPOINT_DIR \
    --num_epochs $NUM_EPOCHS \
    --lr 1e-4 \
    --train_bs 4 \
    --training_seq_len 2048 \
    --save_steps 500 \
    --disable_tqdm True \
    \$RESUME_ARG

echo \"Training checkpoint reached for $MODEL\"
ls -lh $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | tail -5
"
