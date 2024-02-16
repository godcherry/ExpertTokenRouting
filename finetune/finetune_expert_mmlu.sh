#!/bin/bash

PORT=10405
MODEL="Qwen-7B-Chat"
EPOCH=4

categories=("astronomy" "electrical_engineering" "security_studies" "prehistory" "international_law" "human_sexuality")

for category in "${categories[@]}"; do
  args="-m $MODEL -p $PORT -d ./dataset/mmlu/finetune_mmlu.json -o ./expert_models/Qwen_${category} -e $EPOCH"
  echo "Executing with args: $args"
  ./finetune/finetune_ds.sh $args
done