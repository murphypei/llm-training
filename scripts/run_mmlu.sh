#!/usr/bin/env bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=4,5

# general
proj_root=$(dirname $(dirname $0))
timestamp=$(date +%Y-%m-%d_%H%M%S)

# llama2
model_name_or_path=ziqingyang/chinese-alpaca-2-7b
tokenizer_name_or_path=${proj_root}/llama2/tokenizer
output_dir=${proj_root}/llama2/mmlu_result/${timestamp}
mkdir -p ${output_dir}

python evaluation/evaluate_mmlu.py \
    --ngpu 2 \
    --data_dir evaluation/mmlu_test/data \
    --model ${model_name_or_path} \
    --tokenizer ${tokenizer_name_or_path} \
    --save_dir ${output_dir}
