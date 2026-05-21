#!/bin/bash

#SBATCH --job-name=defgen_olmo
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

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

# --train_filename /cluster/projects/nn9851k/andrey/dpo_definitions/periti_2025_data/sft_train.tsv 
# --dev_filename /cluster/projects/nn9851k/andrey/dpo_definitions/periti_2025_data/sft_dev.tsv 
#--output_dir output 
#--per_device_train_batch_size 32 
#--per_device_eval_batch_size 64 
#--num_train_epochs 1 
#--model_name_or_path allenai/Olmo-3-7B-Instruct-SFT 
#--qlora 
#--optim paged_adamw_8bit 
#--max_seq_length 314 
#--context_column example

