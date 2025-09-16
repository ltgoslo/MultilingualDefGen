# inspired by https://www.kaggle.com/code/aisuko/fine-tuning-t5-small-with-lora
import os
from pathlib import Path
import torch
import random
from accelerate import FullyShardedDataParallelPlugin, Accelerator, PartialState
import evaluate
import nltk
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, EarlyStoppingCallback, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from arguments.arguments import ModelArguments, DataTrainingArguments, PEFTArguments


def decode(text):
    try:
        return text.encode().decode('unicode_escape')
    except:
        return text


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
    data_fname = os.path.split(args.train_filename)[-1].split(os.extsep)[0]
    args.tag += data_fname
    output_dir = Path(f'{args.output_dir}/{args.model_name_or_path.strip("/").replace("/", "-")}-{args.tag}')
    logging_dir = str(output_dir.parent) + f'/log_{output_dir.name}'
    

    if args.verbose: print('-- Load train dataset --')
    train_dataset = pd.read_csv(args.train_filename, sep='\t').dropna()
    num_train_samples = train_dataset.shape[0]
    train_dataset = Dataset.from_pandas(train_dataset)
    dev_dataset = pd.read_csv(args.dev_filename, sep='\t')
    if '_de.' in args.dev_filename:
        dev_dataset = dev_dataset.sample(2026, random_state=args.seed, ignore_index=True) # as in Russian, not to waste hours on eval
    dev_dataset = Dataset.from_pandas(dev_dataset)
    num_eval_steps = int(num_train_samples / (args.per_device_train_batch_size * args.gradient_accumulation_steps) / 10)
    print(f"Evaluation every {num_eval_steps} steps")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=num_eval_steps,
        evaluation_strategy="steps",
        eval_steps=num_eval_steps,
        save_strategy="steps",
        save_steps=num_eval_steps,
        push_to_hub=False,
        report_to="wandb",
        overwrite_output_dir=True,
        predict_with_generate=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        optim=args.optim,
        metric_for_best_model="eval_loss",
        save_only_model=True,
        generation_max_length=args.max_target_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_on_start=True,
        group_by_length=True,
    )

    if args.verbose: print('-- Load tokenizer --')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    def preprocess_func(data):
        # tokenize each row of inputs and outputs
        model_inputs = tokenizer(
            data[args.context_column], truncation=True, max_length=args.max_seq_length, padding=False)
        try:
            labels = tokenizer(data[args.gloss_column], truncation=True, max_length=args.max_target_length, padding=False)
        except TypeError:
            for i, defi in enumerate(data[args.gloss_column]):
                if not isinstance(defi, str):
                    print(i, defi) # None happened in Russian dbnary
            raise

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_func, batched=True)
    eval_dataset = dev_dataset.map(preprocess_func, batched=True)
    if args.verbose: print(f'-- Set accelerator --')
    fsdp_plugin = FullyShardedDataParallelPlugin(  # see: https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False))
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    if args.verbose: print(f'-- Set tuning parameters [model, device, cache] --')
    device_index = PartialState().process_index
    device_map = {"": device_index}
    settings = dict(
        pretrained_model_name_or_path=args.model_name_or_path,
        device_map=device_map,
        cache_dir=args.cache_dir,
        )

    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
            bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization
            bnb_4bit_compute_dtype=torch.bfloat16,  # specifies the data type for computation
        )
        settings['quantization_config'] = bnb_config

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo", "lm_head"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    if args.verbose: print(f'-- Load base model --')
    base_model = AutoModelForSeq2SeqLM.from_pretrained(**settings)
    print(base_model)
    base_model.config.use_cache = False  # avoid using cache params
    base_model = prepare_model_for_kbit_training(
        base_model)  # see: https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    base_model.config.pretraining_tp = 1  # info: https://github.com/huggingface/transformers/pull/24906
    model = get_peft_model(base_model, peft_config)
    model = accelerator.prepare_model(model, device_placement=True)
    model.print_trainable_parameters()
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model, 
        padding='longest', 
        max_length=max(args.max_seq_length, args.max_target_length),
        pad_to_multiple_of=None,
        return_tensors='pt',
        )
    if args.verbose: print(f'-- Set Trainer --')

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    rouge_scorer = evaluate.load("rouge")
    bert_scorer = evaluate.load("bertscore")
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
 
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print(trainer.data_collator)
    if args.verbose: print(f'-- Training is started! --')
    trainer.train()
    trainer.model.save_pretrained(str(output_dir) + f'/final-epoch')
    trainer.tokenizer.save_pretrained(str(output_dir) + f'/final-epoch')
    trainer.evaluate()
    pd.DataFrame(trainer.state.log_history).to_csv(str(output_dir) + f'/log.tsv', sep='\t', index=False)
    


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PEFTArguments))
    parser.add_argument('--qlora', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    train(args)