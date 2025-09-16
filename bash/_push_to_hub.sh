#!/bin/bash
#SBATCH -A PROJECT-ID -p alvis
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=A100fat:1

# Initialize global variables
export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0"

# Initialize parameters
peft_model_name_or_path="$1"
model_id="$2"

pretrained_model_name_or_path="Unbabel/TowerInstruct-7B-v0.2"
read_hugginface_token="YOUR-TOKEN[READ]"
write_hugginface_token="YOUR-TOKEN[WRITE]"


python src/push_to_hub.py \
       --read_hugginface_token "${read_hugginface_token}" \
       --write_hugginface_token "${write_hugginface_token}" \
       --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
       --peft_model_name_or_path "${peft_model_name_or_path}" \
       --model_id "${model_id}"
