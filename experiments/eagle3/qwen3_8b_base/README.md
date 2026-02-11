# Eagle3 - Qwen3-8B-Base

Eagle3 draft model training for `Qwen/Qwen3-8B-Base` (base model).

## Pipeline

```bash
sbatch generate_responses.sbatch       # vLLM completion generation
sbatch compute_hidden_states.sbatch    # DP=8, 1 GPU/model
sbatch train.sbatch                    # FSDP2, 8 GPUs, 2 epochs
```