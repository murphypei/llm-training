#!/usr/bin/env bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

# general
proj_root=$(dirname $(dirname $0))
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_name=llama2_pt_${timestamp}
output_dir=${proj_root}/llama2/output/sft_lora/${timestamp}
mkdir -p ${output_dir}

# model
model_name_or_path=ziqingyang/chinese-alpaca-2-7b
tokenizer_name_or_path=${proj_root}/llama2/tokenizer
# peft_model=path/to/peft/model/dir

# training
seed=1234
lr=2e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
max_seq_length=512

# data
data_root=/mnt/cephfs2/peichao/NLP/translate/LLM/datasets/chinese_llama2
dataset_dir=${data_root}/sft_training
validation_file=${data_root}/translate_test_42.json
num_train_epochs=10

# gpu
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=8
deepspeed_config_file=${proj_root}/scripts/ds_config.json

torchrun --nnodes 1 --nproc_per_node 4 ${proj_root}/llama2/run_clm_peft_sft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed ${seed} \
    --fp16 \
    --num_train_epochs ${num_train_epochs} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_steps 300 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --report_to none \
    --run_name ${run_name}
