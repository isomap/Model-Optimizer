"""Aggregate multiple response variants into single conversation entries.

For Eagle3, we generate 32 response variants per prompt. The generation phase
stores each variant as a separate conversation with ID like 'dapo-xxx-r00'.

This script aggregates all variants back into single entries with a proper
structure for efficient hidden state extraction.

This reduces storage by ~32x:
- Before: 32 files × 20 MB = 640 MB per base conversation
- After: 1 file × 20 MB = 20 MB per base conversation

Usage:
    python3 aggregate_responses.py --input conversations_combined.jsonl --output conversations_aggregated.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL with separate response entries")
    parser.add_argument(
        "--output", required=True, help="Output JSONL with aggregated conversations"
    )
    args = parser.parse_args()

    # Read and group by base conversation_id
    conversations = defaultdict(list)
    total_entries = 0
    sentinel_count = 0

    print("Reading input file...")
    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue

            # Skip sentinel lines
            if entry.get("finished"):
                sentinel_count += 1
                continue

            conv_id = entry.get("conversation_id")
            if not conv_id:
                print(f"Warning: Missing conversation_id at line {line_num}", file=sys.stderr)
                continue

            # Extract base ID and response number
            if "-r" in conv_id:
                base_id, response_suffix = conv_id.rsplit("-r", 1)
                try:
                    response_num = int(response_suffix)
                except ValueError:
                    print(
                        f"Warning: Invalid response number '{response_suffix}' at line {line_num}",
                        file=sys.stderr,
                    )
                    base_id = conv_id
                    response_num = 0
            else:
                # No response suffix - treat as base conversation
                base_id = conv_id
                response_num = 0

            conversations[base_id].append((response_num, entry))
            total_entries += 1

    print(f"Read {total_entries} entries ({sentinel_count} sentinels skipped)")
    print(f"Found {len(conversations)} base conversations")

    # Validate and write aggregated conversations
    valid_count = 0
    incomplete_count = 0
    missing_responses = []

    print("Writing aggregated conversations...")
    with open(args.output, "w") as f:
        for base_id, variants in sorted(conversations.items()):
            # Sort by response number
            variants.sort(key=lambda x: x[0])

            # Check if we have all 32 responses (0-31)
            response_nums = [r[0] for r in variants]
            if len(variants) != 32 or response_nums != list(range(32)):
                incomplete_count += 1
                missing_responses.append((base_id, len(variants), response_nums))
                # Still process incomplete conversations
                # continue

            # Get the prompt (everything except the assistant response)
            first_entry = variants[0][1]
            convs = first_entry.get("conversations", [])
            if not convs or len(convs) < 2:
                print(f"Warning: Invalid conversation format for {base_id}", file=sys.stderr)
                continue

            # Extract prompt messages (all except last, which is the assistant response)
            prompt_messages = convs[:-1]

            # Collect all response contents
            responses = []
            for response_num, entry in variants:
                entry_convs = entry.get("conversations", [])
                if not entry_convs or entry_convs[-1].get("role") != "assistant":
                    print(
                        f"Warning: Missing assistant response for {base_id}-r{response_num:02d}",
                        file=sys.stderr,
                    )
                    continue

                assistant_msg = entry_convs[-1]
                responses.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg["content"],
                        "response_id": response_num,
                    }
                )

            # Create ONE aggregated entry with ALL responses
            # This will be processed by a custom hidden state extraction script
            aggregated = {
                "conversation_id": base_id,  # Use base ID without -rXX
                "prompt": prompt_messages,
                "responses": [r["content"] for r in responses],
                "response_ids": [r["response_id"] for r in responses],
                "num_responses": len(responses),
            }

            f.write(json.dumps(aggregated, ensure_ascii=False) + "\n")
            valid_count += 1

    print("\nResults:")
    print(f"  Valid base conversations: {valid_count}")
    print(f"  Total output entries: {valid_count * 32} (approx)")
    if incomplete_count > 0:
        print(f"  Incomplete conversations: {incomplete_count}")
        if len(missing_responses) <= 10:
            for base_id, count, nums in missing_responses[:10]:
                print(f"    {base_id}: {count} responses {nums}")
    print(f"  Written to: {args.output}")

    if valid_count == 0:
        print("ERROR: No valid conversations produced!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
