"""Generate responses via vLLM's OpenAI-compatible API.

Supports both chat completions (instruct models) and plain completions (base models).
Unlike server_generate.py, this always saves the response even if truncated at max_tokens.
"""

import argparse
import concurrent.futures
import json
import os
import sys

import tqdm
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--num_threads", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max_tokens", type=int, default=32000)
parser.add_argument(
    "--max_model_len",
    type=int,
    default=32768,
    help="Model context window for auto-adjusting max_tokens",
)
parser.add_argument("--model", type=str, default="model")
parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
parser.add_argument("--api_key", type=str, default="token-abc123")
parser.add_argument(
    "--chat", action="store_true", help="Use chat completions API (for instruct models)"
)
args = parser.parse_args()

with open(args.data_path) as f:
    data = [json.loads(line) for line in f]

client = OpenAI(base_url=args.url, api_key=args.api_key)

# Resume support: skip already-generated conversations
finished_ids = set()
done = False
if os.path.exists(args.output_path):
    with open(args.output_path) as f:
        for line in f:
            entry = json.loads(line)
            finished_ids.add(entry.get("conversation_id", -1))
            if entry.get("finished", False):
                done = True
                break

output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

if done:
    print("All conversations already generated")
    sys.exit()


def generate(entry):
    cid = entry["conversation_id"]
    prompt = entry["conversations"][0]["content"]
    # Estimate prompt tokens (~4 chars/token) and cap max_tokens to fit context
    est_prompt_tokens = len(prompt) // 3 + 100  # conservative estimate
    max_tokens = min(args.max_tokens, args.max_model_len - est_prompt_tokens)
    max_tokens = max(max_tokens, 256)  # minimum generation length
    try:
        if args.chat:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=args.temperature,
            )
            text = response.choices[0].message.content.strip()
        else:
            response = client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=args.temperature,
            )
            text = response.choices[0].text.strip()

        if not text:
            result = {"conversation_id": cid}
        else:
            result = {
                "conversation_id": cid,
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ],
            }
        with open(args.output_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as e:
        print(f"Error for {cid}: {e}")


with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as pool:
    futures = [pool.submit(generate, e) for e in data if e["conversation_id"] not in finished_ids]
    for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        f.result()

with open(args.output_path, "a") as f:
    f.write(json.dumps({"finished": True}) + "\n")
