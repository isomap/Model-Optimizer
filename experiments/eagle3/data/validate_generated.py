"""Validate and clean generated conversation JSONL files.

Reads a conversations.jsonl, filters to valid entries, and writes
conversations_clean.jsonl for hidden state extraction.

Checks:
- Valid JSON per line
- Has 'conversations' key with >= 2 messages (user + assistant)
- Assistant response is non-empty (> 10 chars)
- Removes the 'finished' sentinel line

Usage:
    python3 validate_generated.py --input conversations.jsonl --output conversations_clean.jsonl
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--min-assistant-chars", type=int, default=10)
    args = parser.parse_args()

    total = 0
    valid = 0
    no_conversations = 0
    too_few_messages = 0
    empty_assistant = 0
    bad_json = 0
    sentinels = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            # Skip sentinel lines
            if entry.get("finished"):
                sentinels += 1
                continue

            convs = entry.get("conversations")
            if convs is None:
                no_conversations += 1
                continue

            if len(convs) < 2:
                too_few_messages += 1
                continue

            # Check assistant response length
            assistant_msg = convs[-1]
            if assistant_msg.get("role") != "assistant":
                too_few_messages += 1
                continue

            content = assistant_msg.get("content", "")
            if len(content) < args.min_assistant_chars:
                empty_assistant += 1
                continue

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            valid += 1

    rejected = total - valid - sentinels
    print(f"Total lines:       {total}")
    print(f"Valid:             {valid}")
    print(f"Rejected:          {rejected}")
    if bad_json:
        print(f"  Bad JSON:        {bad_json}")
    if no_conversations:
        print(f"  No conversations:{no_conversations}")
    if too_few_messages:
        print(f"  Too few msgs:    {too_few_messages}")
    if empty_assistant:
        print(f"  Empty assistant:  {empty_assistant}")
    if sentinels:
        print(f"  Sentinels:       {sentinels}")
    print(f"Written to:        {args.output}")

    if valid == 0:
        print("ERROR: No valid conversations found!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
