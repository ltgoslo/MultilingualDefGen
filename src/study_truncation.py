import pandas as pd
import os
from glob import glob
from transformers import AutoTokenizer

if __name__ == '__main__':
    out_dir = 'token_stats'
    if not os.path.exists(out_dir):
        os.mkdir('token_stats')
    for model in (#"bigscience/mt0-xl",
                  "Unbabel/TowerInstruct-7B-v0.2",
                  ):
        tokenizer = AutoTokenizer.from_pretrained(model)
        for fn in glob(os.path.expanduser("~/defgen_formatted/test*")):
            print(fn)
            fn = os.path.expanduser(fn)
            data = pd.read_csv(fn, sep='\t').dropna()
            with open(os.path.join(out_dir, os.path.split(fn)[-1])+f'-{model.replace("/", "-")}.txt', 'w') as f:
                f.write(fn+'\n')
                f.write(model+'\n')

                if 'Tower' in model:
                    if 'definition' in data.columns:
                        data['toks_example'] = data.apply(lambda x: tokenizer.apply_chat_template(
                                [
                                    {'role': 'user',
                                     'content': x[
                                         'example_with_prompt'
                                     ]},
                                    {'role': 'assistant', 'content': x['definition']}],
                                tokenize=True), axis=1)
                    else:
                        data['toks_example'] = data.apply(
                            lambda x: tokenizer.apply_chat_template(
                                [
                                    {'role': 'user',
                                     'content': x[
                                         'example_with_prompt'
                                     ]},
                                    ],
                                tokenize=True, add_generation_prompt=True), axis=1)
                else:
                    data['toks_example'] = data['example_with_prompt'].apply(tokenizer.tokenize)
                    if 'definition' in data.columns:
                        data['toks_def'] = data['definition'].apply(tokenizer.tokenize)
                        data['toks_def_len'] = data['toks_def'].apply(len)
                        f.write("Def\n")
                        f.write(
                            str(data.toks_def_len.describe()))  # 26 mean Tower, 22 median, 75% 32; t5 mean 22, 19 median, 75% 27
                data['toks_example_len'] = data['toks_example'].apply(len)

                f.write(str(data.shape)+'\n')
                f.write("Example\n")
                f.write(str(data.toks_example_len.describe())) # 58 mean Tower, 52 median, 75% 72; t5 mean 50, 45 median, 75% 62
