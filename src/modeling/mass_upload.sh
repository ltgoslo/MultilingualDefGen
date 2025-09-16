#! /bin/bash

for i in /cluster/projects/nn9851k/mariiaf/definitions/models/*
do
    base_name=$(basename ${i})
    echo ${base_name}
    python3 upload.py ltg/${base_name} ${i}
done