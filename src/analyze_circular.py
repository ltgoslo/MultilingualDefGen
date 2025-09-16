#! /bin/env python3

import sys, statistics
import pandas as pd
import os


def analyze_file(filename):
    lengths = []
    nb_circ = 0
    nb_total = 0

    df = pd.read_csv(filename, sep="\t", header=0)
    for i, row in df.iterrows():
        lengths.append(len(row["gloss"].split(" ")))
        if row["word"] in row["gloss"]:
            nb_circ += 1
        nb_total += 1

    print(f"Circular: {nb_circ}/{nb_total} = {nb_circ/nb_total*100:.2f}%")
    print(f"Average length: {statistics.mean(lengths):.2f}")


if __name__ == "__main__":
    path2scan = sys.argv[1]

    with os.scandir(path2scan) as source:
        for entry in source:
            if entry.name.endswith("labels.tsv.gz") and entry.is_file():
                print("===========")
                print(entry.path)
                analyze_file(entry.path)
                print("===========")
