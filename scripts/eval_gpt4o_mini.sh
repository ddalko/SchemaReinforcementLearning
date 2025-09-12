#!/bin/bash

export CODELINKER_CONFIG="/workspace/SchemaReinforcementLearning/gpt4o-mini.toml"
export CONCURRENCY=1

python schemabench/unionbench.py --model gpt-4o-mini \
       --save_path ./schemabench/results/gpt_4o_mini_schema.jsonl \
       --subset True \
       --test_category schema \
       --n 1