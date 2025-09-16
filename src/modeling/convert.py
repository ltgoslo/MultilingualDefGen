import argparse
import os.path
import re

import pandas as pd

from arguments.constants import TARGET, EXAMPLE, EXAMPLE_WITH_PROMPT, DEFINITION


def replace_prompt(x, pattern):
    match = re.search(pattern, x[EXAMPLE_WITH_PROMPT])
    return {EXAMPLE: x[EXAMPLE_WITH_PROMPT][:match.start()], TARGET: match.group(1)}


def clean_text(text):
    if isinstance(text, str):
        # Handle patterns like #{option1|option2}#
        text = re.sub(r"#\{[^|}]+\|([^}]+)\}#?", r"\1", text)
        # Handle patterns like {option1|option2}# at the start of the sentence
        text = re.sub(r"^\{[^|}]+\|([^}]+)\}#?", r"\1", text)
    return text


prompts = {
        'en': ". What is the definition of <TRG>?",
        'fr': ". Quelle est la définition de <TRG>?",
        'ru': ". Что такое <TRG>?",
        'no': ". Hva betyr <TRG>?",
        'de': ". Was ist die Definition von <TRG>?",
        'fi': ". Mitä tarkoittaa <TRG>?",
        'it': ". Qual è la definizione di <TRG>?",
        'es': ". ¿Cuál es la definición de <TRG>?",
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='dev')
    parser.add_argument('--data_dir', default='/scratch/project_465001925/corpora/defgen_data/')
    parser.add_argument('--out_dir', default='~/defgen_formatted/')
    args = parser.parse_args()
    args.out_dir = os.path.expanduser(args.out_dir)
    args.data_dir = os.path.expanduser(args.data_dir)
    if args.mode == 'test':
        for lang in ('de', 'fi_tarkoittaa', 'ru'):
            data = pd.read_csv(f'{args.data_dir}test_axolotl24st_{lang}.tsv.gz', sep='\t')
            data = data.rename(columns={'Targets': TARGET, "Context": EXAMPLE})
            data[EXAMPLE_WITH_PROMPT] = data.apply(lambda x: x[EXAMPLE].rstrip(
                                ).rstrip('.').rstrip() + prompts[lang].replace(
                                    '<TRG>', x[TARGET]), axis=1)
            data.to_csv(f'{args.data_dir}test_axolotl24st_{lang}.tsv', sep='\t', index=False)
    elif args.mode == 'dev':
        lang = 'fi_tarkoittaa'
        in_ = os.path.join(args.data_dir, f'dev_axolotl24st_{lang}.tsv.gz')
        out = f'{args.out_dir}dev_axolotl24st_{lang}.tsv.gz'
        data = pd.read_csv(in_, sep='\t')
        data = data.rename(columns={EXAMPLE: EXAMPLE_WITH_PROMPT})
        pattern = re.compile(prompts['fi'].replace("<TRG>", "([\w\-]+)\\"))
        applied_df = data.apply(replace_prompt, axis=1, result_type='expand',
                                args=(
                                    pattern,))  # https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
        data = pd.concat([data, applied_df], axis='columns')
        data.to_csv(out, sep='\t', index=False, compression='gzip')
    elif args.mode == 'dbnary':
        data_dbnaries = (
            f"train_axolotl_dbnary_fi_tarkoittaa.tsv.gz",
            f"train_dbnary_de.tsv.gz",
            f"dev_dbnary_de.tsv.gz",
            f"train_axolotl_dbnary_ru.tsv.gz",
        )
        for data_dbnary in data_dbnaries:
            data = pd.read_csv(os.path.join(args.data_dir, data_dbnary), sep='\t')
            data[EXAMPLE] = data[EXAMPLE].apply(clean_text)
            data = data.rename(columns={EXAMPLE: EXAMPLE_WITH_PROMPT})
            data[DEFINITION] = data[DEFINITION].apply(clean_text)
            data.to_csv(
                os.path.join(args.out_dir, data_dbnary),
                sep='\t',
                index=False,
                compression='gzip',
            )
    else:
        for lang in ("fi_tarkoittaa", "ru"):
            data = pd.read_csv(os.path.join(args.data_dir, f"train_axolotl24st_{lang}.tsv.gz"), sep='\t')
            data = data.rename(columns={EXAMPLE: EXAMPLE_WITH_PROMPT})
            data.to_csv(
                os.path.join(args.out_dir, f"train_axolotl24st_{lang}.tsv.gz"),
                sep='\t',
                index=False,
                compression='gzip',
            )