#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH --gpus-per-node=A100fat:1

# Initialize global variables
export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Initialize parameters
language="$1"
peft_model="$2"
model="Unbabel/TowerInstruct-7B-v0.2"
hugginface_token=""
repetition_penalty=1.2
length_penalty=1.2
max_time=7
max_new_tokens=128
batch_size="$3"
checkpoint="$4"
split="$5"


echo "# -- AXOLOTL: $language --"

# Initialize output folder
if [[ "$peft_model" == "None" ]]; then
    folder="TowerInstruct"
elif [[ "$peft_model" == *"dbnary"* ]]; then
    folder="TowerDictionary-${language}_dbnary/$checkpoint"
else
    folder="TowerDictionary-$language/$checkpoint"
fi

# Initialize arrays
inputs=()

axolotl24st_input="train_dev_test_split/${split}_axolotl24st_$language.jsonl"
axolotl24st_output="predictions/${folder}/${split}_axolotl24st_$language.txt"

if [ ! -f "${axolotl24st_output}" ]; then # Finnish and Russian
    inputs+=("$axolotl24st_input")
else
    echo "${axolotl24st_output} already exists"
fi

# Generation
if [[ ${#inputs[@]} -ne 0 ]]; then
    if [[ "$peft_model" == "None" ]]; then
        python src/generation.py \
               --language "$language" \
               --pretrained_model_name_or_path "$model" \
               --peft_model_name_or_path "$peft_model" \
               --quantization \
               --hugginface_token "$hugginface_token" \
               --test_filename "$inputs" \
               --max_new_tokens "$max_new_tokens" \
               --output_dir "predictions/${folder}" \
               --batch_size "$batch_size" \
               --max_time "$max_time" \
               --repetition_penalty "$repetition_penalty" \
               --length_penalty "$length_penalty" \
               --pretrained
    else
        python src/generation.py \
               --language "$language" \
               --pretrained_model_name_or_path "$model" \
               --peft_model_name_or_path "$peft_model" \
               --quantization \
               --hugginface_token "$hugginface_token" \
               --test_filename "$inputs" \
               --max_new_tokens "$max_new_tokens" \
               --output_dir "predictions/${folder}" \
               --batch_size "$batch_size" \
               --max_time "$max_time" \
               --repetition_penalty "$repetition_penalty" \
               --length_penalty "$length_penalty"
    fi
fi
