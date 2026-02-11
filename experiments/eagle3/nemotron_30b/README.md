# Eagle3 - Nemotron-3-Nano-30B-A3B

Eagle3 draft model training for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (instruct).

## Pipeline

```bash
sbatch generate_responses.sbatch       # vLLM chat generation
sbatch compute_hidden_states.sbatch    # DP=4, 2 GPUs/model
sbatch train.sbatch                    # FSDP2, 8 GPUs, 2 epochs
```