#!/bin/bash
#SBATCH -A NAISS2024-22-838 -p alvis
#SBATCH --gpus-per-node=A40:1

### Formatting ###
# test
python src/output_formatting.py --language "ru" --split "test" --output_folder "submissions" --predictions_folder "predictions"
python src/output_formatting.py --language "fi" --split "test" --output_folder "submissions" --predictions_folder "predictions"
python src/output_formatting.py --language "ru" --split "test" --output_folder "submissions" --predictions_folder "predictions" --dbnary
python src/output_formatting.py --language "fi" --split "test" --output_folder "submissions" --predictions_folder "predictions" --dbnary
python src/output_formatting.py --language "de" --split "test" --output_folder "submissions" --predictions_folder "predictions" --dbnary


model="sentence-transformers/distiluse-base-multilingual-cased-v1"


## Russian
# Axolotl
best_checkpoint="checkpoint-279"
gold_data="axolotl24_shared_task/data/russian/axolotl.test.ru.gold.tsv"
data="submissions/TowerDictionary-ru/${best_checkpoint}/test.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

cd "axolotl24_shared_task/code/evaluation/"
mkdir -p ../../../evaluation/submissions/TowerDictionary-ru/${best_checkpoint}/
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

# Dbnary+Axoltl
best_checkpoint="checkpoint-1044"
data="submissions/TowerDictionary-ru_dbnary/${best_checkpoint}/test.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

cd "axolotl24_shared_task/code/evaluation/"
mkdir -p ../../../evaluation/submissions/TowerDictionary-ru_dbnary/${best_checkpoint}/
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

data="submissions/TowerInstruct-ru.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data
cd "axolotl24_shared_task/code/evaluation/"
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

## Finnish
# Axolotl
best_checkpoint="checkpoint-2230"
gold_data="axolotl24_shared_task/data/finnish/axolotl.test.fi.gold.tsv"
data="submissions/TowerDictionary-fi/${best_checkpoint}/test.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

cd "axolotl24_shared_task/code/evaluation/"
mkdir -p ../../../evaluation/submissions/TowerDictionary-fi/${best_checkpoint}/
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

# Dbnary+Axoltl
best_checkpoint="checkpoint-3540"
data="submissions/TowerDictionary-fi_dbnary/${best_checkpoint}/test.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

cd "axolotl24_shared_task/code/evaluation/"
mkdir -p ../../../evaluation/submissions/TowerDictionary-fi_dbnary/${best_checkpoint}/
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

data="submissions/TowerInstruct-fi.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data
cd "axolotl24_shared_task/code/evaluation/"
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

## German
# Dbnary
best_checkpoint="checkpoint-9030"
gold_data="axolotl24_shared_task/data/german/axolotl.test.surprise.gold.tsv"
data="submissions/TowerDictionary-de_dbnary/${best_checkpoint}/test.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data

cd "axolotl24_shared_task/code/evaluation/"
mkdir -p ../../../evaluation/submissions/TowerDictionary-de_dbnary/${best_checkpoint}/
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../

data="submissions/TowerInstruct-de.tsv"
python3 src/sense_label.py --model $model --data $data --bsize 16 --save text --output $data
cd "axolotl24_shared_task/code/evaluation/"
python scorer_track2.py "../../../${data}" "../../../${gold_data}" "../../../evaluation/${data}"
cd ../../../
