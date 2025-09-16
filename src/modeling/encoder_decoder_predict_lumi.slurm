#!/bin/bash

#SBATCH --job-name=defgen_generate
#SBATCH --account=project_465001890
#SBATCH --partition=small-g
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --output=enc-dec-pred-%j.out

source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001925/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

srun singularity exec $SIF python3 encoder_decoder_predict_lumi.py ${@}
