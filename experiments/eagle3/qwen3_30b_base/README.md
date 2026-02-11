# Eagle3 - Qwen3-30B-A3B-Base

Eagle3 draft model training for `Qwen/Qwen3-30B-A3B-Base` (base model, MoE).

## Pipeline

```bash
sbatch generate_responses.sbatch       # vLLM completion generation
sbatch compute_hidden_states.sbatch    # DP=4, 2 GPUs/model
sbatch train.sbatch                    # FSDP2, 8 GPUs, 2 epochs
```