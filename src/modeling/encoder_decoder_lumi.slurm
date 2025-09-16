#!/bin/bash

#SBATCH --job-name=defgen_lora
#SBATCH --account=project_465001925
#SBATCH --partition=small-g
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1


source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001925/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

export WANDB_PROJECT=definition_modeling
srun singularity exec $SIF python3 encoder_decoder_finetuning.py ${@}
