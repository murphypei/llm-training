#!/usr/bin/env bash

set -e
set -x 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
timestamp=$(date +%Y-%m-%d_%H%M%S)

tokenizer=/mnt/cephfs2/peichao/NLP/translate/LLM/Chinese-LLaMA-Alpaca-2/scripts/tokenizer
dataset=/mnt/cephfs2/peichao/NLP/translate/LLM/datasets/chinese_llama2/pt/dedup.json


python pt/chinese_llama2/run_clm.py