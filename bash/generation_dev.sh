# Pretrained
sbatch --time 1:00:00 --job-name RUgeneration --output=slurm_output/generation/TowerInstruc-ru/dev.out bash/_generation.sh "ru" "None" 20 "None" "dev"
sbatch --time 1:00:00 --job-name FIgeneration --output=slurm_output/generation/TowerInstruc-fi/dev.out bash/_generation.sh "fi" "None" 64 "None" "dev"

## Russian
# Axolotl
checkpoints=("checkpoint-31"  "checkpoint-62" "checkpoint-93" "checkpoint-124" "checkpoint-155" "checkpoint-186" "checkpoint-217" "checkpoint-248" "checkpoint-279" "checkpoint-310")
for dir in models/TowerDictionary-ru/checkpoint*; do
    if [ -d "$dir" ]; then
        checkpoint=${dir/models\/TowerDictionary-ru\//}

        if [[ ! " ${checkpoints[@]} " =~ " ${checkpoint} " ]]; then
            continue
        fi

        sbatch --time 1:00:00 --job-name RUgeneration --output=slurm_output/generation/TowerDictionary-ru/${checkpoint}/dev.out bash/_generation.sh "ru" $dir 20 $checkpoint "dev"
    fi
done

# Dbnary+Axolotl
checkpoints=("checkpoint-1044" "checkpoint-2088" "checkpoint-3132" "checkpoint-4176" "checkpoint-5220")
for dir in models/TowerDictionary-ru_dbnary/checkpoint*; do
    if [ -d "$dir" ]; then
        checkpoint=${dir/models\/TowerDictionary-ru_dbnary\//}

        if [[ ! " ${checkpoints[@]} " =~ " ${checkpoint} " ]]; then
            continue
        fi

        sbatch --time 1:00:00 --job-name RUgeneration --output=slurm_output/generation/TowerDictionary-ru_dbnary/${checkpoint}/dev.out bash/_generation.sh "ru" $dir 20 $checkpoint "dev"
    fi
done


## Finnish
# Axolotl
checkpoints=("checkpoint-1338" "checkpoint-1784" "checkpoint-2230" "checkpoint-2676" "checkpoint-3122" "checkpoint-3568" "checkpoint-4014" "checkpoint-446"  "checkpoint-4460" "checkpoint-892")
for dir in models/TowerDictionary-fi/checkpoint*; do
    if [ -d "$dir" ]; then
        checkpoint=${dir/models\/TowerDictionary-fi\//}

        if [[ ! " ${checkpoints[@]} " =~ " ${checkpoint} " ]]; then
            continue
        fi

        sbatch --time 10:00:00 --job-name FIgeneration --output=slurm_output/generation/TowerDictionary-fi/${checkpoint}/dev.out bash/_generation.sh "fi" $dir 30 $checkpoint "dev"
    fi
done

# Dbnart+Axolotl
checkpoints=("checkpoint-1416" "checkpoint-2124" "checkpoint-2832" "checkpoint-3540" "checkpoint-708")
for dir in models/TowerDictionary-fi_dbnary/checkpoint*; do
    if [ -d "$dir" ]; then
        checkpoint=${dir/models\/TowerDictionary-fi_dbnary\//}

        if [[ ! " ${checkpoints[@]} " =~ " ${checkpoint} " ]]; then
            continue
        fi

        sbatch --time 10:00:00 --job-name FIgeneration --output=slurm_output/generation/TowerDictionary-fi_dbnary/${checkpoint}/dev.out bash/_generation.sh "fi" $dir 30 $checkpoint "dev"
    fi
done
