## Russian
# Axoltl
model_path="models/TowerDictionary-ru/checkpoint-279"

sbatch --time=00:40:00 newbash/_push_to_hub.sh "$model_path" "TowerDictionary-ru_axolotl"

# Axoltl+Dbnary
model_path="models/TowerDictionary-ru_dbnary/checkpoint-1044"

sbatch --time=00:40:00 newbash/_push_to_hub.sh "$model_path" "TowerDictionary-ru_axolotl-dbnary"


## Finnish
# Axoltl
model_path="models/TowerDictionary-fi/checkpoint-2230"

sbatch --time=00:40:00 newbash/_push_to_hub.sh "$model_path" "TowerDictionary-fi_axolotl"

# Axoltl+Dbnary
model_path="models/TowerDictionary-fi_dbnary/checkpoint-3540"

sbatch --time=00:40:00 newbash/_push_to_hub.sh "$model_path" "TowerDictionary-fi_axolotl-dbnary"


## German
# Dbnary
model_path="models/TowerDictionary-de_dbnary/checkpoint-9030"

sbatch --time=00:40:00 newbash/_push_to_hub.sh "$model_path" "TowerDictionary-de_dbnary"
