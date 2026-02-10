# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# From https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation
#
import json
import os

from datasets import load_dataset

DATA_PATH = ".../hf_datasets/wikitext-103-v1"
# Load the WikiText-103 dataset
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

# Define the destination folder
os.makedirs(DATA_PATH, exist_ok=True)

# Save splits to JSONL files and calculate their sizes
with open(f"{DATA_PATH}/wikitext-train.jsonl", "w") as file:
    file.writelines(json.dumps(item) + "\n" for item in dataset)

print(f"Raw dataset saved to {DATA_PATH}/wikitext-train.jsonl")
