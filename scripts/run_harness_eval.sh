#!/usr/bin/env bash

set -e
set -x

# general
proj_root=$(dirname $(dirname $0))
timestamp=$(date +%Y-%m-%d_%H%M%S)

# llama2
model_name_or_path=ziqingyang/chinese-alpaca-2-7b
tokenizer_name_or_path=${proj_root}/llama2/tokenizer
output_dir=${proj_root}/llama2/harness_eval_result/${timestamp}
mkdir -p ${output_dir}

python lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=${model_name_or_path},tokenizer=${tokenizer_name_or_path} \
    --tasks hellaswag,piqa,boolq,winogrande,xnli_zh,xcopa_zh,xwinograd_zh,xstory_cloze_zh \
    --device cuda:2 \
    2>&1 | tee $MODEL/harness_eval.log
