#!/bin/bash
wget -O dbnary_ru.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ru_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ru.ttl.bz2
wget -O dbnary_de.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/de_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_de.ttl.bz2
wget -O dbnary_fi.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/fi_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_fi.ttl.bz2

python src/Dbnary.py --filename dbnary_ru.ttl --output data/DBNARY/dbnary_ru.jsonl
python src/Dbnary.py --filename dbnary_de.ttl --output data/DBNARY/dbnary_de.jsonl
python src/Dbnary.py --filename dbnary_fi.ttl --output data/DBNARY/dbnary_fi.jsonl
