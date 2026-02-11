# Eagle3 Speculative Decoding Training

Train Eagle3 draft models for 5 base models using offline training with DAPO-Math-17k dataset.

## Models

| Model | HF ID | Hidden State DP | Training |
|-------|--------|-----------------|----------|
| Qwen3-8B | `Qwen/Qwen3-8B` | DP=8, 1 GPU/model | 1 node, 8 GPU FSDP2 |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | DP=4, 2 GPU/model | 1 node, 8 GPU FSDP2 |
| Qwen3-32B | `Qwen/Qwen3-32B` | DP=4, 2 GPU/model | 1 node, 8 GPU FSDP2 |
| GPT-OSS-20B | `openai/gpt-oss-20b` | DP=4, 2 GPU/model | 1 node, 8 GPU FSDP2 |
| GPT-OSS-120B | `openai/gpt-oss-120b` | DP=1, 8 GPU/model | 1 node, 8 GPU FSDP2 |

## Pipeline

### Phase 1: Data Preparation

```bash
sbatch data/prepare_data.sbatch
```

Converts DAPO-Math-17k (open-r1/DAPO-Math-17k-Processed) into conversation JSONL format.

### Phase 2: Compute Hidden States (per model)

```bash
sbatch qwen3_8b/compute_hidden_states.sbatch
sbatch qwen3_30b/compute_hidden_states.sbatch
# ...etc
```

Runs base model inference to extract hidden states from layers [2, N/2, N-3] for offline training.

### Phase 3: Train Eagle3 (per model)

```bash
sbatch qwen3_8b/train.sbatch
sbatch qwen3_30b/train.sbatch
# ...etc
```

Offline Eagle3 training using pre-computed hidden states. 2 epochs, LR 1e-4, batch size 4, seq len 2048.