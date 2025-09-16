import re
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from trl import SFTTrainer
from huggingface_hub import login
from datasets import load_dataset, Dataset, IterableDataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator, PartialState
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments

def clean_text(text):
    if text.endswith('}.'):
        text=text[:-1]+'#.'

    # Handle patterns like #{option1|option2}#
    text = re.sub(r"#\{[^|}]+\|([^}]+)\}#", r"\1", text)
    # Handle patterns like {option1|option2}# at the start of the sentence
    text = re.sub(r"^\{[^|}]+\|([^}]+)\}#", r"\1", text)
    return text

def formatting_func_factory(tokenizer, args):
    system_message = dict()
    system_message['de'] = "Du bist ein Lexikograf, der mit der Bereitstellung prägnanter Definitionen von Wortbedeutungen vertraut ist."
    system_message['fi'] = "Olet sanakirjantekijä, joka tuntee sanan merkitysten ytimekkäiden määritelmien antamisen."
    system_message['ru'] = "Вы — лексикограф, знакомый с составлением кратких определений значений слов."

    user_message = dict()
    user_message['de'] = 'Bitte geben Sie eine prägnante Definition für die Bedeutung des Wortes "{}" im folgenden Satz an: {}'
    user_message['fi'] = 'Ole hyvä ja anna lyhyt määritelmä sanan "{}" merkitykselle seuraavassa lauseessa: {}'
    user_message['ru'] = 'Пожалуйста, предоставьте краткое определение значения слова "{}" в следующем предложении: {}'

    def formatting_func(record):
       if len(record['definition']) and record['definition'][-1] != '.':
            record['definition']+='.'

        #clean
        record['definition'] = clean_text(record['definition'])
        record['example'] = clean_text(record['example'])
        record['target'] = clean_text(record['target'])

        record['definition']+='<stop>'

        return tokenizer.apply_chat_template([{'role': 'system', 'content': system_message[args.language]},
                                              {'role': 'user', 'content': user_message[args.language].format(record['target'], record['example'])},
                                              {'role': 'assistant', 'content': record['definition']}],
                                             tokenize=False)
    return formatting_func


def train(args):
    if args.verbose: print('-- Set seed --')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.verbose: print('-- Hugginface login --')
    login(args.hugginface_token)


    if args.verbose: print(f'-- Set accelerator --')
    fsdp_plugin = FullyShardedDataParallelPlugin(  # see: https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False))
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


    if args.verbose: print('-- Load tokenizer --')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name,
                                              padding_side="right",
                                              add_eos_token=False,
                                              add_bos_token=False,
                                              cache_dir=args.cache_dir)
    tokenizer.add_tokens(["<stop>"])
    tokenizer.pad_token = tokenizer.eos_token


    if args.verbose: print('-- Load train dataset --')
    if not args.streaming:
        train_dataset = load_dataset('json', data_files=args.train_filename, split='train').shuffle(seed=args.seed).to_pandas()
        train_dataset = Dataset.from_pandas(train_dataset[~train_dataset['example'].isna()])
    else:
        train_dataset = load_dataset('json', data_files=args.train_filename, split='train', streaming=args.streaming).shuffle(seed=args.seed)
        train_dataset = train_dataset.filter(lambda x: x['example'] is not None)

    if args.verbose: print(f'-- Set tuning parameters [model, device, cache] --')
    settings = dict(pretrained_model_name_or_path=args.base_model_name,
                    device_map='auto',
                    cache_dir=args.cache_dir,
                    trust_remote_code=True)

    if args.verbose: print(f'-- QLoRa {"enabled" if args.qlora and args.lora else "disabled"} --')
    if args.qlora and args.lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
            bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # specifies the data type for computation
        )
        settings['quantization_config'] = bnb_config


    if args.verbose: print(f'-- LoRa {"enabled" if args.lora else "disabled"} --')
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "v_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            bias="none",
            task_type="CAUSAL_LM")


    if args.verbose: print(f'-- Load base model --')
    base_model = AutoModelForCausalLM.from_pretrained(**settings)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False  # avoid using cache params
    base_model.gradient_checkpointing_enable()  # this will reduce GPU memory but slow down the process
    base_model = prepare_model_for_kbit_training(base_model)  # see: https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    base_model.config.pretraining_tp = 1  # info: https://github.com/huggingface/transformers/pull/24906
    if not args.lora: model = base_model
    else: model = get_peft_model(base_model, peft_config)
    model = accelerator.prepare_model(model, device_placement=True)


    if args.verbose: print(f'-- Set SFTTrainer --')
    output_dir = Path(f'{args.output_dir}/{args.finetuned_model_name}{args.tag}')
    logging_dir = str(output_dir.parent) + f'/log_{output_dir.name}'
    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        packing=True,
        formatting_func=formatting_func_factory(tokenizer, args),
        train_dataset=train_dataset,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            do_eval=False,
            bf16=True,
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            gradient_accumulation_steps=args.gradient_accumulation_steps, # see: https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260/5
            gradient_checkpointing=True,
            max_steps=args.num_rows if not args.streaming else (args.num_rows // args.batch_size) // args.gradient_accumulation_steps * args.num_train_epochs
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )


    if args.verbose: print(f'-- Training is started! --')
    trainer.train()

    if args.verbose: print(f'-- Store final model --')
    trainer.model.save_pretrained(str(output_dir) + f'/final-epoch{args.num_train_epochs}')
    trainer.tokenizer.save_pretrained(str(output_dir) + f'/final-epoch{args.num_train_epochs}')
    pd.DataFrame(trainer.state.log_history).to_csv(str(output_dir) + f'/log.tsv', sep='\t', index=False)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Finetuning')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--streaming', action='store_true', help='Load a large dataset as IterableDataset')
    parser.add_argument('--base_model_name', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('--hugginface_token', type=str, default='')
    parser.add_argument('--train_filename', type=str, nargs='+', default='data/train.jsonl')
    parser.add_argument('--qlora', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--cache_dir', type=str, default="/mimer/NOBACKUP/groups/cik_data/fra_hf_cache")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetuned_model_name', type=str, default='TowerLanguageBridge-7B')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_rows', type=int, default=-1)
    parser.add_argument('--tag', type=str, default="")
    args = parser.parse_args()
  
    train(args)
