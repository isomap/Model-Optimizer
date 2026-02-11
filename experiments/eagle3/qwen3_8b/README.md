# Eagle3 - Qwen3-8B

Eagle3 draft model training for `Qwen/Qwen3-8B`.

## Hidden State Extraction

DP=8 (1 GPU per model instance, 8 parallel workers). ~1h on 1 node.

```bash
sbatch compute_hidden_states.sbatch
```

## Training

Offline Eagle3 training with FSDP2 on 8 GPUs. 2 epochs.

```bash
sbatch train.sbatch
```