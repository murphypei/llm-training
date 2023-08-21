#!/usr/bin/env bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

# general
proj_root=$(dirname $(dirname $0))
timestamp=$(date +%Y-%m-%d_%H%M%S)

# llama2
model_name_or_path=ziqingyang/chinese-alpaca-2-7b
tokenizer_name_or_path=${proj_root}/llama2/tokenizer
output_dir=${proj_root}/llama2/ceval_result/${timestamp}
mkdir -p ${output_dir}

python evaluation/evaluate_ceval.py \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --output_dir ${output_dir} \
    2>&1 | tee ${output_dir}/eval.log
