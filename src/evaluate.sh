PRED=${1}
GOLD=${2}

python3 modeling/merge_axolotl_definitions.py --definitions_file $PRED --gold $GOLD --replace_definitions
python3 sense_label/sense_label.py --data $PRED".merged.tsv.gz" --model sentence-transformers/all-distilroberta-v1
python3 scorer_track2.py $PRED".merged.tsv.gz.labels.tsv.gz" $GOLD