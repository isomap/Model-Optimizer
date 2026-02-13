#!/bin/bash
# Submit Eagle3 online training for all 7 models with job chaining
# Each model runs 10 epochs across multiple 4-hour jobs

set -eo pipefail

cd "$(dirname "$0")"

MODELS=(
    "qwen3_8b_base"
    "qwen3_8b"
    "qwen3_30b_base"
    "qwen3_30b"
    "qwen3_32b"
    "gpt_oss_20b"
    "gpt_oss_120b"
)

echo "Submitting Eagle3 training for all models..."
echo "Each model will run with job dependencies for automatic resumption"
echo ""

for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    bash "$model/submit_training.sh"
    echo ""
done

echo "All training jobs submitted!"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  ssh dfw1 'tail -f /lustre/fsw/portfolios/coreai/users/\$USER/eagle3/training/*/logs/*.out'"
