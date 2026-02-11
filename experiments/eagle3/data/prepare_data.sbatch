#!/bin/bash
#SBATCH --job-name=eagle3-data-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

LUSTRE_BASE="/lustre/fsw/portfolios/coreai/users/$USER"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/data"
VENV_DIR="$LUSTRE_BASE/eagle3/.venv"
HF_HOME="$LUSTRE_BASE/.cache/huggingface"
export HF_HOME

# Set up venv with datasets
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet datasets tqdm

mkdir -p "$OUTPUT_DIR"

# Load DAPO-Math-17k, deduplicate, expand to 32 responses per prompt
NUM_RESPONSES=32

python -c "
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

NUM_RESPONSES = $NUM_RESPONSES

ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')
print(f'Loaded {len(ds)} rows from DAPO-Math-17k')

# Deduplicate by prompt content hash
seen_hashes = set()
unique_prompts = []
for entry in tqdm(ds, desc='Deduplicating'):
    prompt_msgs = entry.get('prompt', [])
    if not prompt_msgs:
        continue
    content = prompt_msgs[0].get('content', '').strip()
    if not content:
        continue
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    if h in seen_hashes:
        continue
    seen_hashes.add(h)
    unique_prompts.append({'content': content, 'hash': h})

print(f'Found {len(unique_prompts)} unique prompts')

# Expand: 32 copies per prompt with unique IDs for diverse generation at temp=1.0
output_path = Path('$OUTPUT_DIR/dapo_prompts_x${NUM_RESPONSES}.jsonl')
total = 0
with open(output_path, 'w') as f:
    for p in unique_prompts:
        for r in range(NUM_RESPONSES):
            entry = {
                'conversations': [{'role': 'user', 'content': p['content']}],
                'conversation_id': f\"dapo-{p['hash']}-r{r:02d}\",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            total += 1

print(f'Wrote {total} entries ({len(unique_prompts)} prompts x {NUM_RESPONSES} responses) to {output_path}')
"

# Shard for multi-node generation (10k conversations per shard)
SHARD_DIR="$OUTPUT_DIR/shards"
mkdir -p "$SHARD_DIR"
python -c "
import json, os
shard_dir = '$SHARD_DIR'
input_path = '$OUTPUT_DIR/dapo_prompts_x${NUM_RESPONSES}.jsonl'
max_per_shard = 10000
shard_idx = 0
line_count = 0
outfile = open(os.path.join(shard_dir, f'train-{shard_idx:05d}-{shard_idx:05d}.jsonl'), 'w')
with open(input_path) as f:
    for line in f:
        if line_count >= max_per_shard:
            outfile.close()
            shard_idx += 1
            line_count = 0
            outfile = open(os.path.join(shard_dir, f'train-{shard_idx:05d}-{shard_idx:05d}.jsonl'), 'w')
        outfile.write(line)
        line_count += 1
outfile.close()
print(f'Created {shard_idx + 1} shards in {shard_dir}')
"

echo "Data preparation complete."
wc -l "$OUTPUT_DIR/dapo_prompts_x${NUM_RESPONSES}.jsonl"
ls "$SHARD_DIR" | wc -l
echo "shards created"