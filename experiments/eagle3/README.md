# Eagle3 Speculative Decoding Training

Train Eagle3 draft models for 9 base models using offline training with DAPO-Math-17k dataset.

## Models

| Dir | HF ID | Type | HS DP | Gen TP |
|-----|--------|------|-------|--------|
| `qwen3_8b` | `Qwen/Qwen3-8B` | instruct | 8 (1 GPU) | 8 |
| `qwen3_8b_base` | `Qwen/Qwen3-8B-Base` | base | 8 (1 GPU) | 8 |
| `qwen3_30b` | `Qwen/Qwen3-30B-A3B` | instruct | 4 (2 GPU) | 8 |
| `qwen3_30b_base` | `Qwen/Qwen3-30B-A3B-Base` | base | 4 (2 GPU) | 8 |
| `qwen3_32b` | `Qwen/Qwen3-32B` | instruct | 4 (2 GPU) | 8 |
| `gpt_oss_20b` | `openai/gpt-oss-20b` | instruct | 4 (2 GPU) | 8 |
| `gpt_oss_120b` | `openai/gpt-oss-120b` | instruct | 1 (8 GPU) | 8 |
| `nemotron_30b` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | instruct | 4 (2 GPU) | 8 |
| `nemotron_30b_base` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | base | 4 (2 GPU) | 8 |

## Pipeline (4 phases)

### Phase 1: Data Preparation (shared, one-time)

```bash
sbatch data/prepare_data.sbatch
```

Deduplicates `BytedTsinghua-SIA/DAPO-Math-17k` (1.79M rows) to ~17k unique math prompts.

### Phase 2: Generate Responses (per model)

```bash
sbatch qwen3_8b/generate_responses.sbatch
```

Runs vLLM with TP=8 to generate long reasoning chains (up to 32k tokens) for each prompt.
Instruct models use chat mode. Base models use completion mode.

### Phase 3: Compute Hidden States (per model)

```bash
sbatch qwen3_8b/compute_hidden_states.sbatch
```

Extracts hidden states from layers [2, N/2, N-3] for offline Eagle3 training. Max seq len 32768.

### Phase 4: Train Eagle3 (per model)

```bash
sbatch qwen3_8b/train.sbatch
```

Offline Eagle3 training with FSDP2 on 8 GPUs. 2 epochs, LR 1e-4, batch size 4.