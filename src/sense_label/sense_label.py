#!/bin/env python3
# coding: utf-8

import argparse
import logging
import csv
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm

from st_embedding import STEmbedding
from gte_embedding import GTEEmbeddidng


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--data",
        "-d",
        help="Path to the tsv file with definitions, usages and clusters",
        required=True,
    )
    arg(
        "--model",
        "-m",
        help="Path to a Huggingface model",
        default="/fp/projects01/ec30/models"
                "/",
    )  # for English: sentence-transformers/all-distilroberta-v1
    arg("--bsize", "-b", type=int, help="Batch size", default=4)
    arg(
        "--mode",
        type=str,
        choices=["definition", "usage"],
        help="Find most prototypical definition or most prototypical usage?",
        default="definition",
    )
    arg(
        "--save",
        "-s",
        type=str,
        choices=["plot", "text"],
        help="Save plots or usages and definitions?",
        default="text",
    )
    arg(
        "--unique",
        "-u",
        type=int,
        choices=[0, 1],
        help="Try to make sense labels unique (i.e., they do not repeat across senses)",
        default=1,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    args = parse_args()

    if not 'Tower' in args.data:
        dataset = pd.read_csv(args.data, delimiter="\t", quoting=csv.QUOTE_NONE)
    else:
        dataset = pd.read_csv(args.data, delimiter="\t")
    dataset = dataset[dataset.period=='new']
    lemmas = sorted(set(dataset.word.values))

    labels_df = pd.DataFrame({"word": [], "gloss": [], "sense_id": []})

    stats = {}

    if not 'gte' in args.model:
        model = STEmbedding(args.model, args.bsize)
    else:
        model = GTEEmbeddidng(args.model)

    for word in tqdm(lemmas):
        logger.debug(f"Processing {word}...")
        stats[word] = 0
        df = dataset[(dataset.word == word) & (dataset.sense_id != -1)]
        #senses = sorted(set(df.sense_id.values))
        senses = df["sense_id"].value_counts().index
        prototype_definition = ''
        assigned_labels = set()
        for sense in senses:
            definitions = df[(df.sense_id == sense)]["generated_definition"].dropna().tolist()
            if definitions:
                representations = definitions
                proto_markers = [0 for el in range(len(representations))]
                embeddings = model.encode(
                    representations, return_dense=True, return_sparse=True,
                )
                if 'gte' in args.model:
                    embeddings = embeddings['dense_embeddings'].cpu().numpy()

                if len(representations) < 1:
                    logger.debug(f"Sense {sense}:")
                    logger.debug(
                        f"Too few usages/definitions for this sense: {len(representations)}. "
                        f"At least 1 required"
                    )
                    #proto_definitions.append("Too small sense")
                    #prototype_definition = (
                    #    "Too few examples to generate a proper definition!"
                    #)
                else:
                    logger.debug(f"Sense {sense}:")
                    prototype_embedding = np.mean(embeddings, axis=0) # TODO: torch?
                    sims = np.dot(embeddings, prototype_embedding)
                    proto_index = np.argmax(sims)
                    prototype_definition = definitions[proto_index]
                    if args.unique == 1:
                        if len(assigned_labels) > 0:
                            if prototype_definition in assigned_labels:
                                logger.debug(
                                    f"{prototype_definition} rejected because it has already been used for the word {word}"
                                )
                                prototype_definition = None
                                candidates = np.flip(np.argsort(sims))
                                for candidate in candidates:
                                    if definitions[candidate] not in assigned_labels:
                                        prototype_definition = definitions[candidate]
                                if not prototype_definition:
                                    prototype_definition = definitions[proto_index]
                                    logger.debug(
                                        "No other usable definitions found, falling back to the duplicate"
                                    )
                                    stats[word] += 1
                        assigned_labels.add(prototype_definition)
                    logger.debug(prototype_definition)

            labels_df.loc[len(labels_df)] = [
                word,
                prototype_definition,
                sense,
            ]

    if args.unique == 1:
        logger.debug("Non-unique sense labels:")
        non_unique = {(el, stats[el]) for el in stats if stats[el] > 0}
        logger.debug(non_unique)
    output = os.path.expanduser(args.data + '.labels.tsv.gz')
    print(output)
    if not 'Tower' in args.data:
        labels_df.to_csv(
            output,
            sep="\t",
            index=False,
            quoting=csv.QUOTE_NONE,
            compression='gzip',
        )
    else:
        labels_df.to_csv(output, sep="\t", index=False, compression='gzip')
