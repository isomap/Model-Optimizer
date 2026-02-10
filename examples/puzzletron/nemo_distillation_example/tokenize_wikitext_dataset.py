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

from modelopt.torch.utils.plugins import megatron_preprocess_data

DATA_PATH = ".../hf_datasets/wikitext-103-v1"

HF_MODEL_NAME_OR_PATH = ".../puzzle_dir_decilm/ckpts/teacher"

megatron_preprocess_data(
    input_path=f"{DATA_PATH}/wikitext-train.jsonl",
    output_dir=DATA_PATH,
    tokenizer_name_or_path=HF_MODEL_NAME_OR_PATH,
    json_keys=["text"],
    workers=32,
    log_interval=100000,
)
