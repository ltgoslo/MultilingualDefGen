import argparse
import os.path
import re
from collections import Counter

import pandas as pd
import csv

from arguments.constants import EXAMPLE, EXAMPLE_WITH_PROMPT, GENERATED_DEFINITION, TARGET


def replace_prompt(x):
    return re.sub(r" Что такое \w+\?", '',  x[EXAMPLE])

def replace_prompt_fi(x):
    return re.sub(r" \. Mitä tarkoittaa \w+\?", '',  x[EXAMPLE])

def replace_prompt_de(x):
    return re.sub(r"\. Was ist die Definition von \w+\?", '',  x[EXAMPLE])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--definitions_file', default='../../../aya-results/aya-ru-ax.csv')
    parser.add_argument('--replace_definitions', action='store_true')
    parser.add_argument('--gold', default='../../axolotl24_shared_task/data/russian/axolotl.dev.ru.tsv')
    return parser.parse_args()

# '../../../aya-results/aya-ru-ax-db.tsv' # 400, with lower eval rouge


if __name__ == '__main__':
    args = parse_args()
    split = 'test'
    gold = os.path.expanduser(args.gold)
    axolotl = pd.read_csv(gold, sep='\t',quoting=csv.QUOTE_NONE)
    print(axolotl.columns)
    shape_before = axolotl.shape[0]
    print(shape_before)
    # if 'german' in args.gold:
    #     print(axolotl[axolotl[
    #             'example'] == '1. †. Auricularia. der Ohrwurm, Oehrling, Ohrhöhler.'])
    # elif 'finnish' in args.gold:
    #     print(axolotl[axolotl['example']=='ette me sinun lackian henges puhaluxella, ijancaikisesa wighatomisa cukostaijsima'])
    if ('finnish' in args.gold) or ('surprise' in args.gold):
        axolotl = axolotl.drop_duplicates(('word', EXAMPLE))

        print(axolotl.shape)

    args.definitions_file = os.path.expanduser(args.definitions_file)
    if not "Tower" in args.definitions_file:
        definitions = pd.read_csv(args.definitions_file, sep='\t',quoting=csv.QUOTE_NONE)
    else:
        definitions = pd.read_csv(args.definitions_file, sep='\t')
    print(definitions.columns)
    if not EXAMPLE_WITH_PROMPT in definitions.columns:
        if 'ru' in args.definitions_file:
            definitions['example'] = definitions.apply(replace_prompt, axis=1)
        elif 'fi' in args.definitions_file:
            definitions['example'] = definitions.apply(replace_prompt_fi, axis=1)
        elif 'de' in args.definitions_file:
            definitions['example'] = definitions.apply(replace_prompt_de, axis=1)
    if 'surprise' in args.gold:
        source = pd.read_csv(
        f"~/defgen_formatted/test_axolotl24st_de.tsv", sep='\t')
        definitions = definitions[[GENERATED_DEFINITION, TARGET]]
        definitions[EXAMPLE] = source[EXAMPLE]
    else:
        definitions = definitions[[EXAMPLE, GENERATED_DEFINITION, TARGET]]
    definitions[GENERATED_DEFINITION] = definitions[GENERATED_DEFINITION].apply(lambda x: x.strip('"').strip())
    print(definitions.columns)
    print(f"Generated shape: {definitions.shape}")
    print(axolotl[EXAMPLE].unique().shape[0])
    axolotl = axolotl.rename(columns={'word': TARGET})
    #print(Counter(axolotl[EXAMPLE]))
    axolotl = axolotl.merge(definitions, how='left', on=[EXAMPLE, TARGET])
    axolotl = axolotl.rename(columns={TARGET:'word'})
    if args.replace_definitions:
        axolotl['gloss'] = axolotl[GENERATED_DEFINITION]
    print(axolotl.shape[0])
    assert axolotl.shape[0] == shape_before
    out = args.definitions_file + '.merged.tsv.gz'
    print(out)

    if not "Tower" in args.definitions_file:
        axolotl.to_csv(out, sep='\t',quoting=csv.QUOTE_NONE, compression='gzip')
    else:
        axolotl.to_csv(out, sep='\t', compression='gzip')
