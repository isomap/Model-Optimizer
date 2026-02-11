#!/bin/bash
#SBATCH --job-name=eagle3-gen-gpt-oss-120b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j_%a.out
#SBATCH --error=%x_%j_%a.err
#SBATCH --array=0-55

# Each array task processes one shard on its own node with a vLLM server.
# 556k conversations / 10k per shard = 56 shards (indices 0-55).
# Submit with: sbatch --account=<acct> generate_responses.sbatch
# Resubmit safely - server_generate.py skips already-generated conversation IDs.

set -eo pipefail

MODEL="openai/gpt-oss-120b"
MODEL_DIR="gpt_oss_120b"
SHARDS_PER_TASK=1

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
SHARD_DIR="$LUSTRE_BASE/eagle3/data/shards"
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

# Process shard(s) assigned to this array task
SHARD_IDX=$SLURM_ARRAY_TASK_ID
for offset in \$(seq 0 \$(($SHARDS_PER_TASK - 1))); do
    IDX=\$((SHARD_IDX * $SHARDS_PER_TASK + offset))
    SHARD=\$(printf '$SHARD_DIR/train-%05d-%05d.jsonl' \"\$IDX\" \"\$IDX\")
    OUTPUT=\$(printf '$OUTPUT_DIR/output-%05d.jsonl' \"\$IDX\")
    if [ ! -f \"\$SHARD\" ]; then
        echo \"Shard \$SHARD not found, skipping\"
        continue
    fi
    echo \"Processing shard \$IDX: \$SHARD -> \$OUTPUT\"
    python3 $REPO_DIR/experiments/eagle3/data/generate_completions.py \
        --data_path \"\$SHARD\" \
        --output_path \"\$OUTPUT\" \
        --max_tokens 32768 \
        --temperature 1.0 \
        --num_threads 256 \
        --chat
done

echo 'Generation complete for array task $SLURM_ARRAY_TASK_ID'
"
