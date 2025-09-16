import argparse
import os
import pathlib
from collections import defaultdict

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, 
                          set_seed,
                          pipeline,
                          )

from arguments.constants import EXAMPLE_WITH_PROMPT, GENERATED_DEFINITION

MAX_LENGTHS = {
    'fi': 143,
    'de': 360,
    'ru': 562,
}


def get_strategy(args):
    return {
        "greedy": {'num_beams':1, "do_sample": False},
        "contrastive_search": {"penalty_alpha": 0.6, "top_k": 4},
        "multinomial_sampling": {"num_beams": 1, "do_sample": True},
        "beam_search": {"num_beams": 5, "do_sample": False, 'length_penalty': 1.1, "early_stopping": True}, # here and after following https://aclanthology.org/2024.findings-acl.339.pdf
        "beam_search_multinomial_sampling": {"num_beams": 5, "do_sample": True, 'length_penalty': 1.1, "early_stopping": True},
        "diverse_beam_search_decoding": {
            "num_beams": 6, "num_beam_groups":  3, "diversity_penalty": 0.5, 'length_penalty': 1.1, "early_stopping": True,
            }, # i did not found num groups in the paper above, use from hf example
        "dola_decoding": {"dola_layers": "high", "do_sample": False}
    }[args.strategy]


def load_model(args, tokenizer):
    settings = dict(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
        device_map=0,
    )

    if args.quantization:
        settings['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(**settings)
    model.eval()

    if args.peft_model_name_or_path!= "None":
        peft_model = PeftModel.from_pretrained(model, args.peft_model_name_or_path)
        peft_model.eval()
        return peft_model.merge_and_unload()
    else:
        return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        add_eos_token=True,
        add_bos_token=False,
        padding_side="left",

    )
    return tokenizer


def generation(args):
    dataset_pd = pd.read_csv(args.test_filename, sep='\t')
    if args.first_n > 0:
        dataset_pd = dataset_pd.head(args.first_n)
    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer)
    formatting_func = lambda record: tokenizer.apply_chat_template(
        [{'role': 'user', 'content': record}],
        tokenize=True,
        add_generation_prompt=True,
        max_length=MAX_LENGTHS[args.lang],
        truncation=args.truncation,
        padding='max_length',
        return_tensors="pt",
        return_dict=True,
    )
    dataset = dataset_pd['example_with_prompt'].apply(formatting_func).tolist()
    test_iter = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    strategy = get_strategy(args)
    print(strategy)
    stem = pathlib.Path(args.test_filename).stem
    outfile = f'{args.peft_model_name_or_path}/{stem}-min_new_tokens-6-{args.strategy}-Tower-rp_{args.repetition_penalty}-max_new_tokens-{args.max_new_tokens}.tsv'


    with open(outfile, 'w', encoding='utf8') as out:
        for inp  in tqdm(test_iter):
            outputs = model.generate(
                input_ids=inp['input_ids'].squeeze(dim=1).to("cuda:0"),
                attention_mask=inp['attention_mask'].squeeze(dim=1).to("cuda:0"),
                max_new_tokens=60, min_new_tokens=6,repetition_penalty=args.repetition_penalty,
                **strategy,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                out.write(output + '\n')
    print(f"Written to:\n{outfile}")

def format_output(text, args):
    if args.pretrained_model_name_or_path == 'Unbabel/TowerInstruct-7B-v0.2':
        text = text.strip('<|im_start|>user\n')
        gloss = text.split('\n')[-1] # _ is the example with prompt
    if gloss=="": # Error, the model didn't answer
        gloss='.' # This will prevent the evaluation script from failing
    return gloss


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(prog='Generation')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('--peft_model_name_or_path', type=str, default="/cluster/projects/nn9851k/mariiaf/definitions/TowerAxolotl-7B-train_axolotl24st_ru/checkpoint-370/")
    parser.add_argument('--quantization', action='store_true')
    parser.add_argument('--test_filename', type=str, default='/cluster/projects/nn9851k/corpora/defgen_data/test_axolotl24st_ru.tsv.gz')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--cache_dir', default='/fp/projects01/ec403/IN5550/exam/defmod/')
    parser.add_argument('--strategy', default='contrastive_search')
    parser.add_argument('--context_column', default=EXAMPLE_WITH_PROMPT)
    parser.add_argument('--truncation', action='store_true')
    parser.add_argument('--lang', default='fi')
    parser.add_argument('--first_n', default=-1, type=int)
    args = parser.parse_args()
    args.test_filename = os.path.expanduser(args.test_filename)
    print(args)
    generation(args)
