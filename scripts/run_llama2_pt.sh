#!/usr/bin/env bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

# general
proj_root=$(dirname $(dirname $0))
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_name=llama2_pt_${timestamp}
output_dir=${proj_root}/llama2/output/pt_lora/${timestamp}
mkdir -p ${output_dir}

# model
model_name_or_path=ziqingyang/chinese-alpaca-2-7b
tokenizer_name_or_path=${proj_root}/llama2/tokenizer

# training
lr=1e-4
block_size=512
seed=1234
save_steps=100000 
max_train_samples=20000000

# data
train_file=/mnt/cephfs2/peichao/NLP/translate/LLM/datasets/chinese_llama2/pt_training/dedup_10000.json
data_save_dir=${proj_root}/llama2/data_cache/pt_lora
mkdir -p ${data_save_dir} 

# lora
lora_rank=64
lora_alpha=128
lora_dropout=0.05
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
# --modules_to_save ${modules_to_save} \
# --gradient_checkpointing \

# gpu
per_device_train_batch_size=2
per_device_eval_batch_size=1
gradient_accumulation_steps=16
deepspeed_config_file=${proj_root}/scripts/ds_config.json

torchrun --nnodes 1 --nproc_per_node 4 ${proj_root}/llama2/run_clm_peft_pt.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --train_file ${train_file} \
    --data_save_dir ${data_save_dir} \
    --streaming \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed ${seed} \
    --fp16 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps ${save_steps} \
    --max_train_samples ${max_train_samples} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --report_to wandb \
    --run_name ${run_name}
