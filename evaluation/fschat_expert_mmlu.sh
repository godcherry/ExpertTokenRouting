#!/bin/bash

categories=("astronomy" "electrical_engineering" "security_studies" "prehistory" "international_law" "human_sexuality" "origin")
model_port=(21001 21002 21003 21004 21005 21006 21007)
port=(22001 22002 22003 22004 22005 22006 22007)
controller_port=(23001 23002 23003 23004 23005 23006 23007)
gpu=(0 1 2 3 4 5 6)

length=${#categories[@]}

for (( i=0; i<$length; i++ )); do
  export CUDA_VISIBLE_DEVICES=${gpu[$i]}
  python3 -m fastchat.serve.controller --port ${controller_port[$i]} &
  sleep 1
  args="--model-path expert_models/mmlu/Qwen_${categories[$i]} --port ${model_port[$i]} --worker-address http://localhost:${model_port[$i]}  --controller-address http://localhost:${controller_port[$i]} --dtype bfloat16"
  echo "Executing with args: $args"
  python3 -m fastchat.serve.vllm_worker $args &
  python3 -m fastchat.serve.openai_api_server --port ${port[$i]} --controller-address http://localhost:${controller_port[$i]} &
done

wait