# Eagle3 - GPT-OSS-120B

Eagle3 draft model training for `openai/gpt-oss-120b`.

## Hidden State Extraction

DP=1 (8 GPUs for single model instance via device_map=auto). ~4h on 1 node.

```bash
sbatch compute_hidden_states.sbatch
```

## Training

Offline Eagle3 training with FSDP2 on 8 GPUs. 2 epochs.

```bash
sbatch train.sbatch
```