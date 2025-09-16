echo "# -- Dbnary: German --"
python src/train_dev_test.py \
       --train_size 0.85 \
       --test_size 0.10 \
       --seed 42 \
       --dbnary_filename "../KaikkiDictionary/data/DBNARY/dbnary_de.jsonl" \
       --output_folder "train_dev_test_split" \
       --targets_to_exclude "axolotl24_shared_task/data/german/targets.txt"

echo "# -- Dbnary: Finnish --"
python src/train_dev_test.py \
       --train_size 0.85 \
       --test_size 0.10 \
       --seed 42 \
       --dbnary_filename "../KaikkiDictionary/data/DBNARY/dbnary_fi.jsonl" \
       --output_folder "train_dev_test_split" \
       --targets_to_exclude "axolotl24_shared_task/data/finnish/targets.txt"

echo "# -- Dbnary: Russian --"
python src/train_dev_test.py \
       --train_size 0.85 \
       --test_size 0.10 \
       --seed 42 \
       --dbnary_filename "../KaikkiDictionary/data/DBNARY/dbnary_ru.jsonl" \
       --output_folder "train_dev_test_split" \
       --targets_to_exclude "axolotl24_shared_task/data/russian/targets.txt"
