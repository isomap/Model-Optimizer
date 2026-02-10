#!/bin/bash
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
# Based on https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/distillation/distillation.html
#
# NeMo Knowledge Distillation Training Script
# 
# IMPORTANT: This script must be run from the NeMo root directory (where scripts/llm/gpt_train.py exists)
# The checkpoint and config paths below should be absolute paths or relative to the NeMo root directory
#
# Example: If NeMo is at /opt/NeMo and your configs are in /workspace/puzzle_dir_decilm:
#   cd /opt/NeMo
#   export PYTHONPATH=$PYTHONPATH:/workspace/puzzletron/puzzletron/
#   export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer
#   /workspace/puzzle_dir_decilm/global_kd_nemo.sh

# Update these paths to point to your actual checkpoints and data
# Student checkpoint (NeMo 2.0 format) - UPDATE THIS PATH
STUDENT_CKPT=".../puzzle_dir_decilm/ckpts/teacher_nemo"
# Teacher checkpoint (NeMo 2.0 format) - UPDATE THIS PATH (use absolute path or path relative to NeMo root)
TEACHER_CKPT=".../puzzle_dir_decilm/ckpts/teacher_nemo"
# Distillation configuration file - UPDATE THIS PATH (use absolute path or path relative to NeMo root)
DISTILLATION_CONFIG=".../Model-Optimizer/examples/puzzletron/nemo_distillation_example/distill-config.yaml"

# Data paths format: "weight path" where weight is a float (e.g., "1.0 /path/to/data")
# The path should point to the dataset prefix WITHOUT .bin extension (Megatron-LM will look for both .bin and .idx files)
# REQUIRED: Update this with your actual tokenized data path (use absolute path)
DATA_PATHS="1.0 .../hf_datasets/wikitext-103-v1/wikitext-train_text_document"
SEQUENCE_LEN=8192
MICRO_BATCHSIZE=1
GLOBAL_BATCHSIZE=4
STEPS=100

TP=8
CP=1
PP=1
DP=1
NUM_NODES=1
DEVICES_PER_NODE=8

NAME="distill_testrun"
LOG_DIR="./distill_logs/"


launch_cmd="torchrun --nproc_per_node=$(($TP * $CP * $PP * $DP))"

${launch_cmd} scripts/llm/gpt_train.py \
    --name ${NAME} \
    --model_path ${STUDENT_CKPT} \
    --teacher_path ${TEACHER_CKPT} \
    --kd_config ${DISTILLATION_CONFIG} \
    --tp_size ${TP} \
    --cp_size ${CP} \
    --pp_size ${PP} \
    --devices ${DEVICES_PER_NODE} \
    --num_nodes ${NUM_NODES} \
    --log_dir ${LOG_DIR} \
    --max_steps ${STEPS} \
    --gbs ${GLOBAL_BATCHSIZE} \
    --mbs ${MICRO_BATCHSIZE} \
    --data_paths ${DATA_PATHS} \
    --seq_length ${SEQUENCE_LEN}