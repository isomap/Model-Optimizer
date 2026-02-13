"""Extract hidden states for Eagle3 training with efficient storage.

This script is optimized for Eagle3's multi-response format where each base
conversation has 32 response variants. Instead of storing 32 separate files,
it stores all responses' hidden states in a single file, reducing storage by ~32x.

Input format (aggregated):
{
    "conversation_id": "dapo-xxx",
    "prompt": [{role: user, content: ...}, ...],
    "responses": ["response0", "response1", ..., "response31"],
    "response_ids": [0, 1, ..., 31],
    "num_responses": 32
}

Output format (.pt file):
{
    "conversation_id": "dapo-xxx",
    "prompt_input_ids": tensor,
    "responses": [
        {
            "response_id": 0,
            "input_ids": tensor,  # prompt + response
            "hidden_states": tensor,  # output layer
            "aux_hidden_states": tensor  # selected middle layers
        },
        ...  # 31 more
    ]
}

Usage:
    python3 compute_hidden_states_eagle3.py \
        --model Qwen/Qwen3-8B \
        --input-data conversations_aggregated.jsonl \
        --output-dir hidden_states/ \
        --max-seq-len 32768
"""

import argparse
import asyncio
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the HuggingFace model",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help="Maximum sequence length in tokens",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Path to aggregated JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save hidden states",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="Data parallel rank",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=1,
        help="Data parallel world size",
    )
    parser.add_argument(
        "--debug-max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations for debugging",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load data
    dataset = load_dataset("json", data_files=str(args.input_data), split="train")
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    # Shard data
    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    print(f"Sharded to {len(dataset)} conversations for DP#{args.dp_rank}/{args.dp_world_size}")

    # Filter out already processed
    def keep_conversation(entry):
        conv_id = entry["conversation_id"]
        output_file = args.output_dir / f"{conv_id}.pt"
        return not output_file.exists()

    original_num = len(dataset)
    dataset = dataset.filter(keep_conversation)
    print(f"Filtered out {original_num - len(dataset)} already processed conversations")

    if args.debug_max_conversations:
        dataset = dataset.select(range(min(args.debug_max_conversations, len(dataset))))

    # Load model
    model = AutoModel.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_success = 0
    num_skipped_too_long = 0
    num_invalid = 0

    pbar = tqdm(total=len(dataset), desc=f"DP#{args.dp_rank} Processing")

    async def process_conversation(entry):
        nonlocal num_success, num_skipped_too_long, num_invalid, num_hidden_layers

        conv_id = entry["conversation_id"]
        prompt = entry["prompt"]
        responses = entry["responses"]
        response_ids = entry["response_ids"]

        if not prompt or not responses:
            num_invalid += 1
            pbar.update(1)
            return

        response_data = []

        for resp_id, response_text in zip(response_ids, responses):
            # Create full conversation (prompt + this response)
            conversation = prompt + [{"role": "assistant", "content": response_text}]

            # Tokenize
            try:
                tokenizer_output = tokenizer.apply_chat_template(
                    conversation, return_tensors="pt", add_generation_template=False
                )
                # Handle both dict and tensor outputs
                if isinstance(tokenizer_output, dict):
                    input_ids = tokenizer_output["input_ids"]
                else:
                    input_ids = tokenizer_output
            except Exception as e:
                print(f"Tokenization error for {conv_id}-r{resp_id}: {e}")
                continue

            num_tokens = input_ids.shape[1]
            if num_tokens <= 10 or num_tokens > args.max_seq_len:
                num_skipped_too_long += 1
                continue

            # Get hidden states
            with torch.inference_mode():
                outputs = model(input_ids=input_ids.to(model.device), output_hidden_states=True)

                if num_hidden_layers is None:
                    num_hidden_layers = len(outputs.hidden_states) - 1

                # Extract selected layers (2, N/2, N-3) + output layer
                hidden_states = outputs.hidden_states
                selected_indices = sorted(
                    set([2, max(0, num_hidden_layers // 2), max(1, num_hidden_layers - 3)])
                )
                aux_hidden_states = torch.cat(
                    [hidden_states[i].squeeze(0).cpu() for i in selected_indices], dim=-1
                )
                output_hidden_states = hidden_states[-1].squeeze(0).cpu()

            response_data.append(
                {
                    "response_id": resp_id,
                    "input_ids": input_ids.squeeze(0).cpu(),
                    "hidden_states": output_hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                }
            )

        if not response_data:
            num_invalid += 1
            pbar.update(1)
            return

        # Save all responses in ONE file
        output_file = args.output_dir / f"{conv_id}.pt"
        with open(output_file, "wb") as f:
            torch.save(
                {
                    "conversation_id": conv_id,
                    "num_responses": len(response_data),
                    "responses": response_data,
                },
                f,
            )

        num_success += 1
        pbar.update(1)

    async def run_all():
        tasks = [process_conversation(entry) for entry in dataset]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())

    pbar.close()

    print(f"\nDP#{args.dp_rank} Results:")
    print(f"  Successfully processed: {num_success}")
    print(f"  Skipped (too long/short): {num_skipped_too_long}")
    print(f"  Invalid: {num_invalid}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
