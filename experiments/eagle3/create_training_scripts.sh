#!/bin/bash
# Generate online training scripts for all Eagle3 models

set -eo pipefail

# Model configurations: model_dir|hf_model_id|decoder_type|num_runs
MODELS=(
    "qwen3_8b_base|Qwen/Qwen3-8B-Base|llama|3"
    "qwen3_8b|Qwen/Qwen3-8B|llama|3"
    "qwen3_30b_base|Qwen/Qwen3-30B-A3B-Base|llama|4"
    "qwen3_30b|Qwen/Qwen3-30B-A3B|llama|4"
    "qwen3_32b|Qwen/Qwen3-32B|llama|4"
    "gpt_oss_20b|openai/gpt-oss-20b|llama|4"
    "gpt_oss_120b|openai/gpt-oss-120b|llama|5"
)

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_DIR HF_MODEL DECODER NUM_RUNS <<< "$model_config"

    echo "Creating training scripts for $MODEL_DIR..."

    # Create train.sbatch
    cat > "experiments/eagle3/$MODEL_DIR/train.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=eagle3-train-MODEL_DIR
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

MODEL="HF_MODEL"
MODEL_DIR="MODEL_DIR"
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
    --eagle_decoder_type DECODER \
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
EOF

    # Replace placeholders
    sed -i.bak "s|MODEL_DIR|$MODEL_DIR|g" "experiments/eagle3/$MODEL_DIR/train.sbatch"
    sed -i.bak "s|HF_MODEL|$HF_MODEL|g" "experiments/eagle3/$MODEL_DIR/train.sbatch"
    sed -i.bak "s|DECODER|$DECODER|g" "experiments/eagle3/$MODEL_DIR/train.sbatch"
    rm "experiments/eagle3/$MODEL_DIR/train.sbatch.bak"

    # Create submission script
    cat > "experiments/eagle3/$MODEL_DIR/submit_training.sh" << EOF
#!/bin/bash
# Submit chained training jobs for $MODEL_DIR (10 epochs)

set -eo pipefail

ACCOUNT="coreai_horizon_dilations"
NUM_RUNS=$NUM_RUNS

JOB1=\$(sbatch --account=\$ACCOUNT --parsable experiments/eagle3/$MODEL_DIR/train.sbatch)
echo "Submitted job 1: \$JOB1"

PREV_JOB=\$JOB1
for i in \$(seq 2 \$NUM_RUNS); do
    NEXT_JOB=\$(sbatch --account=\$ACCOUNT --dependency=afterok:\$PREV_JOB --parsable experiments/eagle3/$MODEL_DIR/train.sbatch)
    echo "Submitted job \$i: \$NEXT_JOB (depends on \$PREV_JOB)"
    PREV_JOB=\$NEXT_JOB
done

echo ""
echo "Submitted \$NUM_RUNS chained jobs for $MODEL_DIR training"
echo "Monitor with: squeue -u \\$USER"
EOF
    chmod +x "experiments/eagle3/$MODEL_DIR/submit_training.sh"

    echo "  ✓ Created training scripts for $MODEL_DIR ($NUM_RUNS jobs)"
done

echo ""
echo "All training scripts created!"
echo "To submit all models, run: experiments/eagle3/submit_all_training.sh"
