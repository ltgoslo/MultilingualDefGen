import argparse
import os
import pathlib
from collections import defaultdict
import pandas as pd
import torch
from peft import PeftModel, AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    pipeline,
)
import csv
from arguments.constants import (
    EXAMPLE,
    EXAMPLE_WITH_PROMPT,
    GENERATED_DEFINITION,
    TARGET,
)
from add_prompts import clean_text, prompts

MAX_LENGTHS = {
    "fi": 143,
    "de": 360,
    "ru": 562,
    "en": 500,
}


def get_strategy(args):
    return {
        "greedy": {"num_beams": 1, "do_sample": False},
        "contrastive_search": {"penalty_alpha": 0.6, "top_k": 4},
        "multinomial_sampling": {"num_beams": 1, "do_sample": True},
        "beam_search": {
            "num_beams": 5,
            "do_sample": False,
            "length_penalty": 1.1,
            "early_stopping": True,
        },  # here and after following https://aclanthology.org/2024.findings-acl.339.pdf
        "beam_search_multinomial_sampling": {
            "num_beams": 5,
            "do_sample": True,
            "length_penalty": 1.1,
            "early_stopping": True,
        },
        "diverse_beam_search_decoding": {
            "num_beams": 6,
            "num_beam_groups": 3,
            "diversity_penalty": 0.5,
            "length_penalty": 1.1,
            "early_stopping": True,
        },  # i did not found num groups in the paper above, use from hf example
        "dola_decoding": {"dola_layers": "high", "do_sample": False},
    }[args.strategy]


def load_model(args, tokenizer):
    settings = dict(
        pretrained_model_name_or_path=args.model,
        device_map="auto",
    )

    if args.quantization:
        settings["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoPeftModelForCausalLM.from_pretrained(**settings)
    model.eval()

    return model


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        add_eos_token=True,
        add_bos_token=False,
        padding_side="left",
    )
    return tokenizer


def generation(args):
    dataset_pd = pd.read_json(args.test_filename, lines=True)
    if args.first_n > 0:
        dataset_pd = dataset_pd.head(args.first_n)
    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    formatting_func = lambda record: tokenizer.apply_chat_template(
        [{"role": "user", "content": record}],
        tokenize=True,
        add_generation_prompt=False,  # With OLMo3 it doesn't help
        max_length=MAX_LENGTHS[args.lang],
        truncation=args.truncation,
        padding="max_length",
        return_tensors="pt",
        return_dict=True,
    )
    dataset_pd[EXAMPLE] = dataset_pd[EXAMPLE].apply(clean_text)
    dataset_pd[EXAMPLE_WITH_PROMPT] = dataset_pd.apply(
        lambda x: x[EXAMPLE].rstrip().rstrip(".").rstrip()
        + prompts[args.lang].replace("<TRG>", x[TARGET].lower()),
        axis=1,
    )

    dataset = dataset_pd[EXAMPLE_WITH_PROMPT].apply(formatting_func).tolist()
    test_iter = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    strategy = get_strategy(args)
    print(strategy)
    generated_definitions = []
    for inp in tqdm(test_iter):
        inp = inp.to("cuda")
        outputs = model.generate(
            input_ids=inp["input_ids"].squeeze(dim=1),
            attention_mask=inp["attention_mask"].squeeze(dim=1),
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=6,
            repetition_penalty=args.repetition_penalty,
            **strategy,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [el.strip().split("\n")[-1] for el in outputs]
        generated_definitions += outputs
    dataset_pd[GENERATED_DEFINITION] = generated_definitions
    return dataset_pd


def format_output(text, args):
    if args.pretrained_model_name_or_path == "Unbabel/TowerInstruct-7B-v0.2":
        text = text.strip("<|im_start|>user\n")
        gloss = text.split("\n")[-1]  # _ is the example with prompt
    if gloss == "":  # Error, the model didn't answer
        gloss = "."  # This will prevent the evaluation script from failing
    return gloss


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(prog="Generation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument(
        "--test_filename",
        type=str,
        default="english-corpus1.jsonl.gz",
    )
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--strategy", default="contrastive_search")
    parser.add_argument("--truncation", action="store_true")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--first_n", default=-1, type=int)
    args = parser.parse_args()
    args.test_filename = os.path.expanduser(args.test_filename)
    print(args)
    stem = pathlib.Path(args.test_filename).stem
    outfile = f"{args.model.split('/')[-1]}-{stem}-min_new_tokens-6-{args.strategy}-rp_{args.repetition_penalty}-max_new_tokens-{args.max_new_tokens}.tsv.gz"
    result = generation(args)
    result[[TARGET, EXAMPLE, GENERATED_DEFINITION]].to_csv(
        outfile, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL
    )
