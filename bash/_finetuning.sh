#!/bin/bash
#SBATCH -A PROJECT-ID -p alvis
#SBATCH --gpus-per-node=A100fat:1

export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0"

# Arguments
language=$1
dataset=$2

# Parameters
seed=42
dropout=0.1
output_dir="models"
lora_rank=256
lora_alpha=512
batch_size=100
max_seq_length=256
weight_decay=0.01
warmup_ratio=0.15
learning_rate=1e-4
num_train_epochs=5
gradient_accumulation_steps=1
peft_model_name="TowerDictionary"
base_model_name="Unbabel/TowerInstruct-7B-v0.2"
hugginface_token="YOUR-HUGGINFACE-TOKEN-HERE"

# dataset and train epochs
if [[ "$dataset" == "Dbnary+Axolotl" ]]; then
    train_filename=("train_dev_test_split/train_dbnary_$language.jsonl" "train_dev_test_split/train_axolotl24st_$language.jsonl")
    tag="_dbnary"
elif [[ "$dataset" == "Dbnary" ]]; then
    train_filename="train_dev_test_split/train_dbnary_$language.jsonl"
    tag="_dbnary"
elif [[ "$dataset" == "Axolotl" ]]; then
    tag=""
    if [[ "$language" == "ru" ]]; then
        num_train_epochs=50
    elif [[ "$language" == "fi" ]]; then
        num_train_epochs=10
    fi
    train_filename="train_dev_test_split/train_axolotl24st_$language.jsonl"
fi

echo "# -- Dbnary: $language --"
python src/finetuning.py \
       --language $language \
       --base_model_name $base_model_name \
       --hugginface_token $hugginface_token \
       --train_filename $train_filename \
       --lora \
       --qlora \
       --lora_rank $lora_rank \
       --lora_alpha $lora_alpha \
       --lora_dropout $dropout \
       --cache_dir $TMPDIR \
       --seed $seed \
       --finetuned_model_name "$peft_model_name-$language" \
       --output_dir $output_dir \
       --max_seq_length $max_seq_length \
       --verbose \
       --weight_decay $weight_decay \
       --warmup_ratio $warmup_ratio \
       --batch_size $batch_size \
       --learning_rate $learning_rate \
       --num_train_epochs $num_train_epochs \
       --gradient_accumulation_steps $gradient_accumulation_steps \
       --num_rows -1 \
       --tag $tag
