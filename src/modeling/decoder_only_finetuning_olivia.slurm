#!/bin/bash

#SBATCH --job-name=defgen_olmo
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=2

source ${HOME}/.bashrc

# Set proxy settings for HTTP and HTTPS traffic
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

echo "--Node: $(hostname)"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_arm_nlpl.sif"

export WANDB_PROJECT=defgen_olmo

export SINGULARITYENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF nvidia-smi
apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 decoder_only_finetuning.py ${@}

# --train_filename /cluster/projects/nn9851k/andrey/dpo_definitions/periti_2025_data/sft_train_prompts.tsv.gz
# --dev_filename /cluster/projects/nn9851k/andrey/dpo_definitions/periti_2025_data/sft_dev_prompts.tsv.gz
#--output_dir output
#--per_device_train_batch_size 32
#--per_device_eval_batch_size 64
#--num_train_epochs 50
#--model_name_or_path allenai/Olmo-3-7B-Instruct-SFT
#--lora
#--max_seq_length 300
#--weight_decay 0.01
#--learning_rate 1e-4
#--warmup_ratio 0.15

