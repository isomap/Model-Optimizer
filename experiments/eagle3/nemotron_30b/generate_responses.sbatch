#!/bin/bash
#SBATCH --job-name=eagle3-gen-nemotron-30b
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
# Resubmit safely - generate_completions.py skips already-generated conversation IDs.

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_DIR="nemotron_30b"
CHAT_FLAG="--chat"

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
REPO_DIR="$LUSTRE_BASE/ghq/github.com/isomap/Model-Optimizer"
SHARD_DIR="$LUSTRE_BASE/eagle3/data/shards"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/generated/$MODEL_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
CONTAINER="vllm/vllm-openai:v0.12.0"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

IDX=$(printf "%05d" "$SLURM_ARRAY_TASK_ID")
SHARD="$SHARD_DIR/train-${IDX}-${IDX}.jsonl"
OUTPUT="$OUTPUT_DIR/output-${IDX}.jsonl"
LOG="$LOG_DIR/task-${IDX}.log"

if [ ! -f "$SHARD" ]; then
    echo "Shard $SHARD not found, exiting"
    exit 0
fi

# Write inner script to avoid bash -c quoting issues
INNER_SCRIPT="$LOG_DIR/run-${IDX}.sh"
cat > "$INNER_SCRIPT" << INNER
#!/bin/bash
set -eo pipefail
export HF_HOME=$LUSTRE_BASE/.cache/huggingface

echo "Installing Mamba dependencies for Nemotron"
pip cache purge 2>/dev/null || true
pip install --no-cache-dir causal_conv1d mamba_ssm

echo "Starting vLLM server for $MODEL"
vllm serve $MODEL \\
    --tensor-parallel-size 8 \\
    --served-model-name model \\
    --port 8000 --host 0.0.0.0 \\
    --trust-remote-code \\
    --max-model-len 32768 &

echo "Waiting for vLLM health..."
for i in \$(seq 1 120); do
    resp=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo 000)
    [ "\$resp" = "200" ] && echo "Server ready after \${i}x5s" && break
    sleep 5
done

# Abort if server never became ready
resp=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo 000)
if [ "\$resp" != "200" ]; then
    echo "ERROR: vLLM server failed to start after 10 minutes. Aborting."
    exit 1
fi

pip install --quiet openai tqdm

echo "Processing shard $SHARD -> $OUTPUT"
python3 $REPO_DIR/experiments/eagle3/data/generate_completions.py \\
    --data_path $SHARD \\
    --output_path $OUTPUT \\
    --max_tokens 32768 \\
    --temperature 1.0 \\
    --num_threads 256 \\
    $CHAT_FLAG

echo "Done. Output lines: \$(wc -l < $OUTPUT 2>/dev/null || echo 0)"
INNER
chmod +x "$INNER_SCRIPT"

srun --container-image="$CONTAINER" \
     --container-mounts="$LUSTRE_BASE:$LUSTRE_BASE" \
     --container-workdir="$REPO_DIR" \
     bash "$INNER_SCRIPT" > "$LOG" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID complete. Log: $LOG"