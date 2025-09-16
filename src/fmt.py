import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='ru')
    parser.add_argument('--data_path', default='~/aya-results/test-long/bigscience-mt0-xl-defmodtrain_axolotl24st_fi_tarkoittaa-rpenalty-1.1-sampling-0-filter-0-contr-False.results.tsv.gz')
    args = parser.parse_args()
    data_path = os.path.expanduser(args.data_path)
    examples_with_prompts, definitions = [], []
    with open(data_path, 'r') as f:
        data = f.read()
        splt = data.split('\n<|im_start|> user')
        for sample in splt:
            splt_2 = sample.split('\n')
            example_with_prompt, definition = splt_2[1], splt_2[3]
            examples_with_prompts.append(example_with_prompt)
            definitions.append(definition)
    result = pd.DataFrame({"real_example_with_prompt": examples_with_prompts, "generated_definition": definitions})
    source = pd.read_csv(f"~/defgen_formatted/test_axolotl24st_{args.lang}.tsv", sep='\t')
    resulting = pd.concat((source, result), axis=1)
    res_path = data_path + "no-trunc.tsv.gz"
    resulting.to_csv(res_path, sep='\t', compression='gzip', index=False)
    print(res_path)
