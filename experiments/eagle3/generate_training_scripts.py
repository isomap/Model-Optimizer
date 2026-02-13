#!/usr/bin/env python3
"""Generate online training scripts for all Eagle3 models."""

from pathlib import Path

# Model configurations
MODELS = [
    ("qwen3_8b_base", "Qwen/Qwen3-8B-Base", "llama", 3),
    ("qwen3_8b", "Qwen/Qwen3-8B", "llama", 3),
    ("qwen3_30b_base", "Qwen/Qwen3-30B-A3B-Base", "llama", 4),
    ("qwen3_30b", "Qwen/Qwen3-30B-A3B", "llama", 4),
    ("qwen3_32b", "Qwen/Qwen3-32B", "llama", 4),
    ("gpt_oss_20b", "openai/gpt-oss-20b", "llama", 4),
    ("gpt_oss_120b", "openai/gpt-oss-120b", "llama", 5),
]

TRAIN_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eagle3-train-{model_dir}
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

MODEL="{hf_model}"
MODEL_DIR="{model_dir}"
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

srun --container-image="$CONTAINER" \\
     --container-mounts="$MOUNTS" \\
     --container-workdir="$REPO_DIR" \\
     bash -c "
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface
export TOKENIZERS_PARALLELISM=False

pip install --quiet --no-deps -e .
pip install --quiet accelerate datasets 'transformers>=4.51' 'huggingface_hub>=0.24.0'

cd examples/speculative_decoding

# Check for existing checkpoint to resume
RESUME_ARG=\\\"\\\"
if [ -d \\"$CHECKPOINT_DIR\\" ] && [ \\\"\\$(ls -A $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | wc -l)\\\" -gt 0 ]; then
    LATEST_CKPT=\\$(ls -td $CHECKPOINT_DIR/checkpoint-* | head -1)
    echo \\\"Resuming from: \\$LATEST_CKPT\\\"
    RESUME_ARG=\\\"--resume_from_checkpoint \\$LATEST_CKPT\\\"
fi

# Create FSDP config
cat > fsdp_config.json << 'FSDP_EOF'
{{
  \\"fsdp_transformer_layer_cls_to_wrap\\": [\\"LlamaDecoderLayer\\", \\"Qwen2DecoderLayer\\"],
  \\"fsdp_backward_prefetch\\": \\"backward_pre\\",
  \\"fsdp_state_dict_type\\": \\"full_state_dict\\",
  \\"fsdp_auto_wrap_policy\\": \\"TRANSFORMER_BASED_WRAP\\",
  \\"fsdp_use_orig_params\\": true
}}
FSDP_EOF

bash launch_train.sh \\
    --model $MODEL \\
    --data $DATA_PATH \\
    --mode eagle3 \\
    --eagle_decoder_type {decoder} \\
    --output_dir $CHECKPOINT_DIR \\
    --num_epochs $NUM_EPOCHS \\
    --lr 1e-4 \\
    --train_bs 4 \\
    --training_seq_len 2048 \\
    --save_steps 500 \\
    --disable_tqdm True \\
    \\$RESUME_ARG

echo \\\"Training checkpoint reached for $MODEL\\\"
ls -lh $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | tail -5
"
"""

SUBMIT_SCRIPT_TEMPLATE = """#!/bin/bash
# Submit chained training jobs for {model_dir} (10 epochs)

set -eo pipefail

ACCOUNT="coreai_horizon_dilations"
NUM_RUNS={num_runs}

JOB1=$(sbatch --account=$ACCOUNT --parsable experiments/eagle3/{model_dir}/train.sbatch)
echo "Submitted job 1: $JOB1"

PREV_JOB=$JOB1
for i in $(seq 2 $NUM_RUNS); do
    NEXT_JOB=$(sbatch --account=$ACCOUNT --dependency=afterok:$PREV_JOB --parsable experiments/eagle3/{model_dir}/train.sbatch)
    echo "Submitted job $i: $NEXT_JOB (depends on $PREV_JOB)"
    PREV_JOB=$NEXT_JOB
done

echo ""
echo "Submitted $NUM_RUNS chained jobs for {model_dir} training"
echo "Monitor with: squeue -u \\$USER"
"""


def main():
    base_dir = Path("experiments/eagle3")

    for model_dir, hf_model, decoder, num_runs in MODELS:
        model_path = base_dir / model_dir

        # Create train.sbatch
        train_script = TRAIN_SBATCH_TEMPLATE.format(
            model_dir=model_dir, hf_model=hf_model, decoder=decoder
        )
        (model_path / "train.sbatch").write_text(train_script)

        # Create submit_training.sh
        submit_script = SUBMIT_SCRIPT_TEMPLATE.format(model_dir=model_dir, num_runs=num_runs)
        submit_file = model_path / "submit_training.sh"
        submit_file.write_text(submit_script)
        submit_file.chmod(0o755)

        print(f"✓ {model_dir}: train.sbatch + submit_training.sh ({num_runs} jobs)")

    print(f"\nCreated training scripts for {len(MODELS)} models")


if __name__ == "__main__":
    main()
