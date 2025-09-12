#!/bin/bash

export CODELINKER_CONFIG="/workspace/SchemaReinforcementLearning/llada.toml"
export CONCURRENCY=1

MODEL_NAME=llada_8b_base

python schemabench/unionbench.py --model $MODEL_NAME \
       --save_path ./schemabench/results/$MODEL_NAME_schema.jsonl \
       --subset True \
       --test_category schema \
       --n 1
