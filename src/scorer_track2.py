#!/usr/bin/env python3
# coding: utf-8

import argparse
import itertools
import evaluate
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU
import torch

# Scoring script to evaluate submissions to Track 1 of the AXOLOTL'24 shared task
# https://github.com/ltgoslo/axolotl24_shared_task/

def infinify(array, i, j):
    array[i, :] = -float("inf")
    array[:, j] = -float("inf")

p = argparse.ArgumentParser()
p.add_argument("submission")
p.add_argument("reference")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sub = pd.read_csv(args.submission, sep="\t")
sub = sub.fillna("")
ref = pd.read_csv(args.reference, sep="\t")
old_senses = set(ref[ref.period == "old"].sense_id.unique())
ref = ref[~ref.sense_id.apply(old_senses.__contains__)].reset_index()

sub = sub[~sub.sense_id.apply(old_senses.__contains__)].reset_index()  # sanity check
assert len(sub) != 0, "no (usable) predictions"
verbose = True
apply_iou_penalty = False
normalize_penalty = False

words_sub = set(sub.word.unique())
words_ref = set(ref.word.unique())
if len(words_sub & words_ref) == 0:
    scores = {
        "bleu": 0,
        "bertscore": 0,
        "IoU": 0,
        "delta": ref.groupby("word").size().mean(),
    }
else:
    # wrongfully included / omitted words penalty
    iou = len(words_sub & words_ref) / len(words_sub | words_ref)
    coverage = len(words_sub & words_ref) / len(words_ref)
    bertscorer = evaluate.load("bertscore")
    bleuscorer = BLEU(effective_order=True, smooth_method='exp')  # default smoothing, but I'm being explicit
    words = words_sub & words_ref
    records = []
    for word in words:
        pred_novel_senses = sub[sub.word == word].gloss.to_list()
        n_pred = len(pred_novel_senses)
        true_novel_senses = ref[ref.word == word].gloss.unique().tolist()
        n_true = len(true_novel_senses)
        sense_preds, sense_refs = zip(
            *itertools.product(pred_novel_senses, true_novel_senses)
        )
        bert_scores = bertscorer.compute(
                predictions=sense_preds,
                references=sense_refs,
                model_type="bert-base-multilingual-cased",
                device=device,
            )
        all_bert_scores = np.array(bert_scores["f1"]).reshape(n_pred, n_true)
        all_bert_precisions = np.array(bert_scores["precision"]).reshape(n_pred, n_true)
        all_bert_recalls = np.array(bert_scores["recall"]).reshape(n_pred, n_true)
        alignments = []
        summed_bertscores = 0.0
        summed_bertprecisions = 0.0
        summed_bertrecalls = 0.0
        for runs in range(min(n_pred, n_true)):
            i, j = np.unravel_index(all_bert_scores.argmax(), all_bert_scores.shape)
            alignments.append((i, j))
            summed_bertscores += all_bert_scores[i, j]
            summed_bertprecisions += all_bert_precisions[i, j]
            summed_bertrecalls += all_bert_recalls[i, j]
            for array in (all_bert_scores, all_bert_precisions, all_bert_recalls):
                infinify(array, i, j)
        summed_bleuscores = 0.0
        for i, j in alignments:
            bleuscore = bleuscorer.sentence_score(
                pred_novel_senses[i], [true_novel_senses[j]]
            )
            summed_bleuscores += bleuscore.score
        norm = max(n_true, n_pred) if normalize_penalty else len(alignments)
        records.append(
            {
                "bleu": (summed_bleuscores / 100) / norm,
                "bertscore": summed_bertscores / norm,
                "bertscore_p": summed_bertprecisions / norm,
                "bertscore_r": summed_bertrecalls / norm,
                "IoU": iou,
                "Coverage (target words)": coverage,
                "delta": abs(n_true - n_pred),
                "word": word,
            }
        )

    scores_df = pd.DataFrame.from_records(records)
    scores_df.to_csv(args.submission+'.sample.csv', index=False)
    scores = {col: scores_df[col].mean() for col in scores_df.columns[:-1]}
    if apply_iou_penalty:
        scores["bleu"] *= iou
        scores["bertscore"] *= iou

if verbose:
    import pprint
    pprint.pprint(scores)
    pprint.pprint("Reference target words with new senses:")
    print(sorted(words_ref))
    pprint.pprint("Submitted target words with new senses:")
    print(sorted(words_sub))

with open(args.submission + ".scores", "w", encoding="utf8") as ostr:
    print(f"BLEU: {scores['bleu']:0.4f}", file=ostr)
    print(f"BERTScore: {scores['bertscore']:0.4f}", file=ostr)
    print(f"BERTScore precision: {scores['bertscore_p']:0.4f}", file=ostr)
    print(f"BERTScore recall: {scores['bertscore_r']:0.4f}", file=ostr)
