#!/bin/bash
#SBATCH --job-name=eagle3-gen-qwen3-8b-base
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

MODEL="Qwen/Qwen3-8B-Base"
MODEL_DIR="qwen3_8b_base"

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
INPUT_DATA="$LUSTRE_BASE/eagle3/data/dapo_prompts_x32.jsonl"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/generated/$MODEL_DIR"

CONTAINER="vllm/vllm-openai:v0.8.5"
MOUNTS="$LUSTRE_BASE:$LUSTRE_BASE"

srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$REPO_DIR" \
     bash -c "
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface
mkdir -p $OUTPUT_DIR

# Base model: serve without chat template, use completions API
vllm serve $MODEL \
    --tensor-parallel-size 8 \
    --served-model-name model \
    --port 8000 --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 32768 &

echo 'Waiting for vLLM server...'
while true; do
    resp=\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health || true)
    [ \"\$resp\" -eq 200 ] && echo 'Server ready' && break
    sleep 10
done

pip install --quiet openai tqdm
python3 $REPO_DIR/experiments/eagle3/data/generate_completions.py \
    --data_path $INPUT_DATA \
    --output_path $OUTPUT_DIR/conversations.jsonl \
    --max_tokens 32000 \
    --temperature 1.0 \
    --num_threads 256

echo 'Generation complete'
wc -l $OUTPUT_DIR/conversations.jsonl
"