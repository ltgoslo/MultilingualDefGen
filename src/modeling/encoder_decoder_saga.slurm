#!/bin/bash

# Job name:
#SBATCH --job-name=enc-dec-axolotl
#
# Project:
#SBATCH --account=nn9851k
#
# Wall time limit:
#SBATCH --time=10:00:00
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=enc-dec-axolotl-%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module --force swap StdEnv Zen2Env
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-numpy/1.24.4-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
module load nlpl-wandb/0.15.2-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

module list    # For easier debugging

export WANDB_PROJECT=huggingface
export WANDB__SERVICE_WAIT=300
python "encoder_decoder_finetuning.py"  ${@}
