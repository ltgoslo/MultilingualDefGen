# clone repository
git clone https://github.com/ltgoslo/axolotl24_shared_task.git

# Download DE test set
wget https://zenodo.org/records/8197553/files/dwug_de_sense.zip?download=1
unzip dwug_de_sense.zip\?download\=1
rm dwug_de_sense.zip\?download\=1
mv dwug_de_sense axolotl24_shared_task/data/german
cd dwug_de_sense axolotl24_shared_task/data/german
python surprise.py --dwug_path dwug_de_sense/ # it will not work without slash!

# Process Axolotl24st datasets
python src/Axolotl24st_formatting.py
