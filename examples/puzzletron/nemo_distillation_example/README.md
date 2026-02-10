# NeMo Distillation Example

## Docker Container Setup

```bash
submit_job --partition interactive --time 4 --image /.../docker/nemo_25_07_liana_puzzletron_distillation_guide.sqsh --mounts /...:/workspace --interactive --gpu
```

## HuggingFace to NeMo Conversion

Convert HuggingFace model to NeMo format:

```bash
python -m export.MCore.convert_puzzletron_hf_to_nemo_with_api \
    --input-ckpt-path /workspace/puzzle_dir_decilm/ckpts/teacher/ \
    --output-ckpt-path /workspace/puzzle_dir_decilm/ckpts/teacher_nemo
```
