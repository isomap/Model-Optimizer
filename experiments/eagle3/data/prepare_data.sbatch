#!/bin/bash
#SBATCH --job-name=eagle3-data-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=experiments/eagle3/data/logs/%j.out
#SBATCH --error=experiments/eagle3/data/logs/%j.err

set -eo pipefail

LUSTRE_BASE="/lustre/fsw/portfolios/general/users/$USER"
REPO_DIR="$LUSTRE_BASE/repos/Model-Optimizer"
OUTPUT_DIR="$LUSTRE_BASE/eagle3/data"
HF_HOME="$LUSTRE_BASE/.cache/huggingface"
export HF_HOME

# Clone or update repo
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/NVIDIA/Model-Optimizer.git "$REPO_DIR"
else
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"

# Install dependencies
pip install -e ".[hf]"
pip install datasets tqdm

mkdir -p "$OUTPUT_DIR"

# Prepare DAPO conversations into JSONL format
python -c "
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset('open-r1/DAPO-Math-17k-Processed', 'en', split='train')
output_path = Path('$OUTPUT_DIR/dapo.jsonl')
conversations = []

for entry in tqdm(ds, desc='Processing DAPO-Math-17k'):
    conv_id = entry.get('extra_info', {}).get('index', '')
    if not conv_id:
        continue
    prompt = entry.get('prompt', '').strip()
    solution = str(entry.get('solution', '')).strip()
    if not prompt or not solution:
        continue
    conversations.append({
        'conversation_id': f'dapo-{conv_id}',
        'conversations': [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': solution},
        ],
    })

with open(output_path, 'w') as f:
    for c in conversations:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')

print(f'Wrote {len(conversations)} conversations to {output_path}')
"

echo "Data preparation complete."
wc -l "$OUTPUT_DIR/dapo.jsonl"