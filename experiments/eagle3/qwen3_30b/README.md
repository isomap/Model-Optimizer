# Eagle3 - Qwen3-30B-A3B

Eagle3 draft model training for `Qwen/Qwen3-30B-A3B`.

## Hidden State Extraction

DP=4 (2 GPUs per model instance via device_map=auto, 4 parallel workers). ~2h on 1 node.

```bash
sbatch compute_hidden_states.sbatch
```

## Training

Offline Eagle3 training with FSDP2 on 8 GPUs. 2 epochs.

```bash
sbatch train.sbatch
```