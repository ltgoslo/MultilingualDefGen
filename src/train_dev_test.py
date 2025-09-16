import re
import json
import string
import pathlib
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning)

def fix_parentheses(text):
    if text.startswith("("):
        return text

    # Check if there is a pattern where a closing parenthesis is followed by any text and then an opening parenthesis
    if re.search(r"\)[^(]*\(", text):
        return "(" + text  # Add an opening parenthesis at the start

    # If the text contains a closing parenthesis before an opening one, prepend an opening parenthesis
    if ")" in text and "(" not in text:
        return "(" + text  # Add an opening parenthesis at the start

    return text

def clean(text):
    # remove \n characters
    text = text.replace('\n', '; ').strip()

    # remove punctuation as a last character
    if len(text) > 1 and text[-1] in string.punctuation:
        text = text[:-1]

    # remove double spaces
    text = " ".join(text.split()).strip()

    # remove punctuation as a first character
    if len(text) > 1 and text[0] in string.punctuation:
        text = text[1:]

    # uppercase first letter
    if len(text) > 1:
        text = text[0].upper() + text[1:]

    return fix_parentheses(text)


def train_dev_test_split(df, args):
    if args.targets_to_exclude:
        targets_to_exclude = [t.strip() for t in open(args.targets_to_exclude, mode='r', encoding='utf-8')]
    else:
        targets_to_exclude = []

    df = df[~df['target'].isin(targets_to_exclude)]
    targets = df['target'].unique()

    train_targets, remaining_targets = train_test_split(targets, test_size=1-args.train_size, random_state=args.seed)
    dev_targets, test_targets = train_test_split(remaining_targets, test_size=args.test_size / (1-args.train_size), random_state=args.seed)
    train_df = df[df['target'].isin(train_targets)]
    test_df = df[df['target'].isin(test_targets)]
    dev_df = df[df['target'].isin(dev_targets)]
    return train_df, dev_df, test_df


def load_data(args, filename):
    records = list()
    for record in open(filename, mode='r', encoding='utf-8'):
        record = json.loads(record)
        if not len(record['examples']): continue

        record['definition'] = clean(record['definition'])

        if len(record['definition']) == '': continue

        for example in record['examples']:
            example = clean(example)
            records.append(dict(target=record['target'], definition=record['definition'], example=example))

            for variant in record['variants']:

                if example.count(record['target']) == 1:
                    records.append(dict(target=variant, definition=record['definition'], example=example.replace(record['target'], variant)))

    return pd.DataFrame(records)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train-dev-test split")
    parser.add_argument('--train_size', type=float, default=0.75)
    parser.add_argument('--test_size', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dbnary_filename', type=str, default='data')
    parser.add_argument('--output_folder', type=str, default='data')
    parser.add_argument('--targets_to_exclude', type=str, default='', help='Filename containing a list of targets to exclude')
    args = parser.parse_args()

    df = load_data(args, args.dbnary_filename)
    train, dev, test = train_dev_test_split(df, args)

    stem = pathlib.Path(args.dbnary_filename).stem
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    train.to_json(f'{args.output_folder}/train_{stem}.jsonl', orient='records', lines=True)
    dev.to_json(f'{args.output_folder}/dev_{stem}.jsonl', orient='records', lines=True)
    test.to_json(f'{args.output_folder}/test_{stem}.jsonl', orient='records', lines=True)
