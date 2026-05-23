#!/bin/bash

#SBATCH --job-name=defgen_olmo
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1

source ${HOME}/.bashrc

# Set proxy settings for HTTP and HTTPS traffic
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

echo "--Node: $(hostname)"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_arm_nlpl.sif"

export SINGULARITYENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF nvidia-smi
apptainer exec --nv -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 decoder_only_predict_olivia.py ${@}

# --model output/3epochs/allenai-Olmo-3-7B-Instruct-SFT-defmodsft_train_prompts/final-epoch 
# --test_filename test.jsonl 
# --batch_size 32 
# --strategy greedy
