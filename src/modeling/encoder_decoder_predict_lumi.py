#!/bin/env python3
# coding: utf-8
import argparse
import logging
import os

import pandas as pd
import torch
import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

from arguments.constants import EXAMPLE_WITH_PROMPT, GENERATED_DEFINITION, TARGET

def define(
        in_prompts,
        lm,
        cur_tokenizer,
        arguments,
        targets,
        filter_target=False,
        num_beams=1,
        num_beam_groups=1,
        sampling=False,
        temperature=1.0,
        repetition_penalty=1.0,
        contrastive=False,
):
    logger.info(f"Tokenizing with max length {arguments.maxl}...")
    inputs = cur_tokenizer(
        in_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=arguments.maxl,
    )
    logger.info("Tokenizing finished.")

    target_ids = cur_tokenizer(targets, add_special_tokens=False).input_ids
    target_ids = torch.tensor([el[-1] for el in target_ids])

    if torch.cuda.is_available():
        inputs = inputs.to("cuda:0")
        target_ids = target_ids.to("cuda:0")

    test_dataset = torch.utils.data.TensorDataset(
        inputs["input_ids"], inputs["attention_mask"], target_ids
    )
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=arguments.bsize, shuffle=False
    )
    logger.info(f"Generating definitions with batch size {arguments.bsize}...")
    gen_args = dict(
        do_sample=sampling,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    if contrastive:
        gen_args["penalty_alpha"] = 0.6
        gen_args["top_k"] = 4
    if num_beam_groups > 1:
        gen_args["diversity_penalty"] = 0.5
    if args.num_beams > 1:
        gen_args["early_stopping"] = True
        gen_args["length_penalty"] = 1.1
    definitions = []
    with torch.autocast("cuda"):
        for inp, att, targetwords  in tqdm.tqdm(test_iter):
            if filter_target:
                bad = [[el] for el in targetwords.tolist()]
                outputs = lm.generate(
                    input_ids=inp,
                    attention_mask=att,
                    max_new_tokens=60,
                    bad_words_ids=bad,
                    **gen_args,
            )
            else:
                outputs = lm.generate(
                    input_ids=inp, attention_mask=att, max_new_tokens=60, **gen_args
            )
            predictions = cur_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            definitions += predictions
    logger.info(f"Generating definitions finished")
    return definitions


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model", "-m", help="Path to or name of a model", required=True)
    arg(
        "--cache_dir",
         default="/fp/projects01/ec30/mariiaf/peft_defmod/",
         help="Where to load the base model from",
        )
    arg(
        "--test_filename",
        "-t",
        help="Path to the file with the input data",
        required=True,
    )
    arg("--bsize", "-b", type=int, help="Batch size", default=4)
    arg("--maxl", "-ml", type=int, help="Max source length", default=256)
    arg(
        "--filter",
        "-f",
        type=int,
        help="Filter out target word from definitions?",
        choices=[0, 1],
        default=0,
    )
    arg(
        "--sampling",
        "-smpl",
        type=int,
        help="Sampling instead of greedy decoding",
        choices=[0, 1],
        default=0,
    )
    arg("--rpenalty", "-rep", type=float, help="Repetition penalty", default=1.0)
    arg(
        "--num_beams",
        "-beams",
        type=int,
        help="Number of beams for beam search",
        default=1,
    )
    arg(
        "--num_beam_groups",
        "-bg",
        type=int,
        help="Number of beam groups for beam search",
        default=1,
    )
    arg("--context_column", default=EXAMPLE_WITH_PROMPT)
    arg("--contrastive", action='store_true')
    args = parser.parse_args()
    args.save = f"{args.model}rpenalty-{args.rpenalty}-sampling-{args.sampling}-filter-{args.filter}-contr-{args.contrastive}.results.tsv.gz"
    args.test_filename = os.path.expanduser(args.test_filename)
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
        bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # specifies the data type for computation
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir=args.cache_dir,
    )

    logger.info(f"Model loaded from {args.model}")

    test_dataframe = pd.read_csv(args.test_filename, sep='\t')

    answers = define(
        test_dataframe[args.context_column].tolist(),
        model,
        tokenizer,
        args,
        test_dataframe[TARGET].tolist(),
        filter_target=args.filter,
        sampling=args.sampling,
        repetition_penalty=args.rpenalty,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        contrastive=args.contrastive,
    )
    print(answers[0])
    test_dataframe[GENERATED_DEFINITION] = answers
    
    test_dataframe.to_csv(
        args.save, sep="\t", encoding="utf-8", compression='gzip',
    )
    logger.info(f"Predictions saved to: \n{args.save}")
