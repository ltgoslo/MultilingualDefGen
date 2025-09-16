import pathlib
import argparse
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train-dev-test split")
    parser.add_argument('--axolotl_folder', type=str, default='axolotl24_shared_task')
    parser.add_argument('--output_folder', type=str, default='train_dev_test_split')
    args = parser.parse_args()

    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)


    targets = defaultdict(list)
    for split in ['train', 'dev', 'test']:
        for lang, language in {'fi': 'finnish', 'ru': 'russian'}.items():
            df = pd.read_csv(f'{args.axolotl_folder}/data/{language}/axolotl.{split}.{lang}.{"gold." if split=="test" else ""}tsv', sep='\t')
            df = df[['word', 'gloss', 'example']].rename(columns={'word':'target', 'gloss':'definition'})
            df.to_json(f'{args.output_folder}/{split}_axolotl24st_{lang}.jsonl', orient='records', lines=True)

            if split in ['test', 'dev']:
                targets[language].extend(df.target.unique())

    for lang in targets:
        open(f'{args.axolotl_folder}/data/{lang}/targets.txt', mode='w', encoding='utf-8').writelines("\n".join(targets[lang]))


    lang, language="de", "german"
    df = pd.read_csv(f'{args.axolotl_folder}/data/{language}/axolotl.test.surprise.gold.tsv', sep='\t')
    df = df[['word', 'gloss', 'example']].rename(columns={'word':'target', 'gloss':'definition'})
    df.to_json(f'{args.output_folder}/test_axolotl24st_de.jsonl', orient='records', lines=True)
    open(f'{args.axolotl_folder}/data/german/targets.txt', mode='w', encoding='utf-8').writelines("\n".join(df.target.unique()))
