import os
import random
from pathlib import Path

import evaluate
import nltk
import numpy as np
import pandas as pd
import torch
from accelerate import (Accelerator, FullyShardedDataParallelPlugin)
from arguments.arguments import (DataTrainingArguments, ModelArguments,
                                 PEFTArguments)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, EarlyStoppingCallback,
                          HfArgumentParser, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def train(args):
    if args.verbose: print('-- Set seed --')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.cache_dir = os.path.expanduser(args.cache_dir)
    args.include_for_metrics = ["loss"] # to get it from compute_metrics
    args.train_filename = os.path.expanduser(args.train_filename)
    args.dev_filename = os.path.expanduser(args.dev_filename)
    print(args)

    if args.verbose: print(f'-- Set accelerator --')
    # should make sense for multi-gpu setup. Not tested on multi-gpu
    fsdp_plugin = FullyShardedDataParallelPlugin(
        # see: https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp
        state_dict_config=FullStateDictConfig(offload_to_cpu=False,
                                              rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False,
                                                         rank0_only=True))
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    if args.verbose: print('-- Load tokenizer --')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=False,
        cache_dir=args.cache_dir,
    )

    def formatting_func(record):
        return tokenizer.apply_chat_template(
            [
                {'role': 'user',
                 'content': record[args.context_column]},
                {'role': 'assistant', 'content': record[args.gloss_column]}],
            tokenize=False)

    if args.verbose: print(
        f'-- Set tuning parameters [model, device, cache] --')
    settings = dict(
        pretrained_model_name_or_path=args.model_name_or_path,
        device_map='auto',
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        )

    if args.verbose: print(
        f'-- QLoRa {"enabled" if args.qlora and args.lora else "disabled"} --')
    if args.qlora and args.lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_use_double_quant=True,
            # enables double quantization (speed-up finetuning)
            bnb_4bit_quant_type="nf4",
            # specifies the type of 4-bit quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
            # specifies the data type for computation
        )
        settings['quantization_config'] = bnb_config

    if args.verbose: print(
        f'-- LoRa {"enabled" if args.lora else "disabled"} --')
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], # all Linear modules
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.verbose: print(f'-- Load base model --')
    base_model = AutoModelForCausalLM.from_pretrained(**settings)
    print(base_model) # to know targer modules for LoRA: use all Linear layers
    
    base_model.config.use_cache = False  # avoid using cache params
    base_model.gradient_checkpointing_enable()  # this will reduce GPU memory but slow down the process
    base_model = prepare_model_for_kbit_training(
        base_model)  # see: https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    base_model.config.pretraining_tp = 1  # info: https://github.com/huggingface/transformers/pull/24906
    if not args.lora:
        model = base_model
    else:
        model = get_peft_model(base_model, peft_config)
    model = accelerator.prepare_model(model, device_placement=True)

    if args.verbose: print(f'-- Set SFTTrainer --')
    data_fname = os.path.split(args.train_filename)[-1].split(os.extsep)[0]
    args.tag += data_fname
    output_dir = Path(
        f'{args.output_dir}/{args.model_name_or_path.strip("/").replace("/", "-")}-{args.tag}'
    )
    logging_dir = str(output_dir.parent) + f'/log_{output_dir.name}'
    train_dataset = pd.read_csv(args.train_filename, sep='\t').dropna()
    num_train_samples = train_dataset.shape[0]
    train_dataset = Dataset.from_pandas(train_dataset)
    dev_dataset = pd.read_csv(args.dev_filename, sep='\t').dropna()
    dev_dataset = Dataset.from_pandas(dev_dataset)
    num_eval_steps = int(num_train_samples / (args.per_device_train_batch_size * args.gradient_accumulation_steps) / 10)
    print(f"Evaluation every {num_eval_steps} steps")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in the predictions as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        rouge_scores = rouge_scorer.compute(predictions=decoded_preds, references=decoded_labels, tokenizer=lambda x: x.split())
        bert_scores = bert_scorer.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                model_type="bert-base-multilingual-cased",
                device=device,
            )
        result = {k: round(v * 100, 4) for k, v in rouge_scores.items()}
        result["BERTScore_precision"] = round((sum(bert_scores["precision"])/len(bert_scores["precision"])) * 100, 4)
        result["BERTScore_recall"] = round((sum(bert_scores["recall"])/len(bert_scores["recall"])) * 100, 4)
        result["BERTScore_f1"] = round((sum(bert_scores["f1"])/len(bert_scores["f1"])) * 100, 4)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    rouge_scorer = evaluate.load("rouge")
    bert_scorer = evaluate.load("bertscore")
    response_template_with_context = "<|im_start|>assistant"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)  
    trainer_args = SFTConfig(
            output_dir=output_dir,
            logging_dir=logging_dir,
            logging_strategy=args.eval_strategy,
            logging_steps=num_eval_steps,
            evaluation_strategy=args.eval_strategy,
            eval_steps=num_eval_steps,
            save_strategy=args.eval_strategy,
            save_steps=num_eval_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            do_eval=True,
            bf16=True,
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            optim=args.optim,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # see: https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260/5
            gradient_checkpointing=True,
            max_seq_length=args.max_seq_length,
            metric_for_best_model="eval_rouge1",
            save_only_model=True,
            load_best_model_at_end=True, # for EarlyStoppingCallback
            eval_accumulation_steps=1,
            bf16_full_eval=True,
            eval_on_start=True,
            save_total_limit=4,
        )
    print(trainer_args)
    trainer = SFTTrainer(
        model=model,
        formatting_func=formatting_func,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        args=trainer_args,
        data_collator=DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits, dim=-1), # This allows us to reduce the size of the logits stored on the GPU https://discuss.huggingface.co/t/cuda-out-of-memory-during-evaluation-but-training-is-fine/1783
    )
    if args.verbose: print(f'-- Training is started! --')
    trainer.train()
    trainer.model.save_pretrained(str(output_dir) + f'/final-epoch')
    trainer.tokenizer.save_pretrained(str(output_dir) + f'/final-epoch')
    trainer.evaluate()
    pd.DataFrame(trainer.state.log_history).to_csv(
        str(output_dir) + f'/log.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PEFTArguments))
    parser.add_argument('--qlora', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    train(args)