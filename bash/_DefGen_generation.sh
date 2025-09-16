#!/bin/bash
#SBATCH -A project-id -p alvis
#SBATCH --gpus-per-node=A100fat:1

# Initialize global variables
export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0"

# Initialize parameters
language="$1"
peft_model="$2"
model="Unbabel/TowerInstruct-7B-v0.2"
hugginface_token=""
repetition_penalty=1.2
length_penalty=1.2
max_time=5
max_new_tokens=128
batch_size="$3"

echo "# -- Dbnary: $language --"

# Initialize output folder
if [[ "$peft_model" == "None" ]]; then
    folder="TowerInstruct"
else
    folder="TowerDictionary-$language"
fi

# Initialize arrays
inputs=()

if [[ "$language" == "fi" || "$language" == "ru" || "$language" == "de" ]]; then
    axolotl24st_test_input="train_dev_test_split/test_axolotl24st_$language.jsonl"
    axolotl24st_test_output="predictions/${folder}/test_axolotl24st_$language.txt"

    if [ ! -f "${axolotl24st_test_output}" ]; then
        inputs+=("$axolotl24st_test_input")
    else
        echo "- Skipping: $axolotl24st_test_output already exists."
    fi
fi

dbnary_test_input="train_dev_test_split/test_dbnary_$language.jsonl"
dbnary_test_output="predictions/${folder}/test_dbnary_$language.txt"

if [ ! -f "${dbnary_test_output}" ]; then
    inputs+=("$dbnary_test_input")
else
    echo "- Skipping: $dbnary_test_output already exists."
fi

if [[ ${#inputs[@]} -ne 0 ]]; then
    python src/DefGen_generation.py \
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
