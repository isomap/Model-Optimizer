# Eagle3 Data Preparation

Prepare DAPO-Math-17k conversations into JSONL format for hidden state extraction.

## Usage

```bash
sbatch prepare_data.sbatch
```

Output: `dapo.jsonl` containing conversations in `{conversation_id, conversations}` format.