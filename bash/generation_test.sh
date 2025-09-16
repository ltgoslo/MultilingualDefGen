## Russian
# Axolotl
best_checkpoint="checkpoint-279"
dir="models/TowerDictionary-ru/${best_checkpoint}"
sbatch --time 1:00:00 --job-name RUgeneration --output=slurm_output/generation/TowerDictionary-ru/"${best_checkpoint}"/test.out bash/_generation.sh "ru" $dir 20 $best_checkpoint "test"

# Dbnary+Axoltl
best_checkpoint="checkpoint-1044"
dir="models/TowerDictionary-ru_dbnary/${best_checkpoint}"
sbatch --time 1:00:00 --job-name RUgeneration --output=slurm_output/generation/TowerDictionary-ru_dbnary/"${best_checkpoint}"/test.out bash/_generation.sh "ru" $dir 20 $best_checkpoint "test"

# Pretrained
sbatch --time 1:00:00 --job-name RUgeneration --output=slurm_output/generation/TowerInstruc-ru/test.out bash/_generation.sh "ru" "None" 20 "None" "test"


## Finnish
# Axolotl
best_checkpoint="checkpoint-2230"
dir="models/TowerDictionary-fi/${best_checkpoint}"
sbatch --time 6:00:00 --job-name FIgeneration --output=slurm_output/generation/TowerDictionary-fi/"${best_checkpoint}"/test.out bash/_generation.sh "fi" $dir 20 $best_checkpoint "test"

# Dbnary+Axoltl
best_checkpoint="checkpoint-3540"
dir="models/TowerDictionary-fi_dbnary/${best_checkpoint}"
sbatch --time 6:00:00 --job-name FIgeneration --output=slurm_output/generation/TowerDictionary-fi_dbnary/"${best_checkpoint}"/test.out bash/_generation.sh "fi" $dir 20 $best_checkpoint "test"

# Pretrained
sbatch --time 6:00:00 --job-name FIgeneration --output=slurm_output/generation/TowerInstruc-fi/test.out bash/_generation.sh "fi" "None" 20 "None" "test"

## German
best_checkpoint="checkpoint-9030"
dir="models/TowerDictionary-de_dbnary/${best_checkpoint}"
sbatch --time 3:00:00 --job-name DEgeneration --output=slurm_output/generation/TowerDictionary-de_dbnary/"${best_checkpoint}"/test.out bash/_generation.sh "de" $dir 20 $best_checkpoint "test"

# Pretrained
sbatch --time 3:00:00 --job-name DEgeneration --output=slurm_output/generation/TowerInstruc-de/test.out bash/_generation.sh "de" "None" 20 "None" "test"
