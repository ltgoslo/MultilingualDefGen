#!/bin/bash
#SBATCH -A NAISS2024-22-838 -p alvis
#SBATCH --gpus-per-node=A40:1

### Formatting ###
# dev
python src/output_formatting.py --language "ru" --split "dev" --output_folder "submissions" --predictions_folder "predictions"
python src/output_formatting.py --language "fi" --split "dev" --output_folder "submissions" --predictions_folder "predictions"
python src/output_formatting.py --language "ru" --split "dev" --output_folder "submissions" --predictions_folder "predictions" --dbnary
python src/output_formatting.py --language "fi" --split "dev" --output_folder "submissions" --predictions_folder "predictions" --dbnary


model="sentence-transformers/distiluse-base-multilingual-cased-v1"


## Russian
# Axolotl
gold_data="axolotl24_shared_task/data/russian/axolotl.dev.ru.tsv"
for checkpoint in "submissions"/TowerDictionary-ru/*; do
    data="$checkpoint/dev.tsv"
    python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

    cd "axolotl24_shared_task/code/evaluation/"
    mkdir -p ../../../evaluation/$checkpoint/
    python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
    cd ../../../
done

# Dbnary+Axoltl
for checkpoint in "submissions"/TowerDictionary-ru_dbnary/*; do
    data="$checkpoint/dev.tsv"
    python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

    cd "axolotl24_shared_task/code/evaluation/"
    mkdir -p ../../../evaluation/$checkpoint/
    python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
    cd ../../../
done


## Finnish
# Axolotl
gold_data="axolotl24_shared_task/data/finnish/axolotl.dev.fi.tsv"
for checkpoint in "submissions"/TowerDictionary-fi/*; do
    data="$checkpoint/dev.tsv"
    python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

    cd "axolotl24_shared_task/code/evaluation/"
    mkdir -p ../../../evaluation/$checkpoint/
    python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
    cd ../../../
done

# Dbnary+Axoltl
for checkpoint in "submissions"/TowerDictionary-fi_dbnary/*; do
    data="$checkpoint/dev.tsv"
    python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

    cd "axolotl24_shared_task/code/evaluation/"
    mkdir -p ../../../evaluation/$checkpoint/
    python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
    cd ../../../
done
