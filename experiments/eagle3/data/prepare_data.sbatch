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