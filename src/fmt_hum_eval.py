import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lpath', default='~/Downloads/test-long/CohereLabs-aya-101-defmodtrain_axolotl24st_fi_tarkoittaa.results.tsv.gz.merged.tsv.gz.labels.tsv.gz')
    parser.add_argument('--gold', default='~/PycharmProjects/axolotl24_shared_task/data/finnish/axolotl.test.fi.gold.tsv')
    args = parser.parse_args()
    print(args)
    labels = pd.read_csv(os.path.expanduser(args.lpath), sep='\t')
    shape_before = labels.shape[0]
    print(labels.shape)
    golds_file = os.path.expanduser(args.gold)
    golds = pd.read_csv(golds_file, sep='\t')

    glosses = golds.drop_duplicates('sense_id')
    # glosses = glosses[glosses.period=='new']
    scores_path = args.lpath + '.sample.csv'
    scores = pd.read_csv(scores_path)
    labels = labels.merge(glosses, how='left', on=('sense_id', 'word'))
    assert labels.shape[0] == shape_before
    labels = labels.merge(scores, how='left', on='word')
    labels = labels[
        ['word', 'gloss_x', 'gloss_y', 'sense_id', 'orth', 'bleu', 'bertscore',
         'bertscore_p', 'bertscore_r']]
    labels.dropna(inplace=True) # drop not-novel senses

    labels = labels.rename(columns={'gloss_x': 'generated_definition',
                                    'gloss_y': 'gold_definition'})
    out = args.lpath + '_hum_eval.tsv.gz'
    print(out)
    labels.to_csv(out, sep='\t', compression='gzip', index=False)
