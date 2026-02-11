# Eagle3 - GPT-OSS-120B

Eagle3 draft model training for `openai/gpt-oss-120b` (instruct, MoE).

## Pipeline

```bash
sbatch generate_responses.sbatch       # vLLM chat generation
sbatch compute_hidden_states.sbatch    # DP=1, 8 GPUs/model
sbatch train.sbatch                    # FSDP2, 8 GPUs, 2 epochs
```