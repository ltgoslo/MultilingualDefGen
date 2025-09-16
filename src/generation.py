import time
import torch
import pathlib
import argparse
from peft import PeftModel
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

def load_model(args, tokenizer):
    login(args.hugginface_token)
    settings = dict(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    device_map='auto')

    if args.quantization:
        settings['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(**settings)

    if not args.pretrained:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    if args.peft_model_name_or_path!= "None":
        peft_model = PeftModel.from_pretrained(model, args.peft_model_name_or_path)
        peft_model.eval()
        return peft_model.merge_and_unload()
    else:
        return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              padding_side="left",
                                              #use_fast=False,
                                              add_eos_token=True,
                                              add_bos_token=True)
    if not args.pretrained:
        tokenizer.add_tokens(["<stop>"])
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
        return tokenizer.apply_chat_template([{'role': 'system', 'content': system_message[args.language]},
                                              {'role': 'user', 'content': user_message[args.language].format(record['target'], record['example'])}],
                                               tokenize=False, add_generation_prompt=True)
    return formatting_func

def generation(pipe, dataset, args):
    formatting_func = formatting_func_factory(pipe.tokenizer, args)
    prompts = [formatting_func(row) for row in dataset]

    tokens = ['.', ' .']
    if not args.pretrained:
        tokens.append('<stop>')
    eos_tokens = [pipe.tokenizer.eos_token_id] + [pipe.tokenizer.encode(token, add_special_tokens=False)[0]
                                                  for token in tokens]

    outputs = list()
    start_time = time.time()
    for out in pipe(prompts,
                         forced_eos_token_id = eos_tokens,
                         max_time = args.max_time * args.batch_size,
                         repetition_penalty = args.repetition_penalty,
                         length_penalty = args.length_penalty,
                         eos_token_id = eos_tokens,
                         max_new_tokens = args.max_new_tokens,
                         truncation = True,
                         batch_size = args.batch_size,
                         pad_token_id = pipe.tokenizer.eos_token_id):
        outputs.append(format_output(out, args))
    print('Generation time:', time.time() - start_time)
    print('# inputs:', len(outputs))
    return outputs

def format_output(text, args):
    if args.pretrained_model_name_or_path == 'Unbabel/TowerInstruct-7B-v0.2':
        if not args.pretrained:
            text = text[0]["generated_text"].split('<|im_start|>assistant\n')[1].split('<stop>')[0]
            # remove newline characters and double spaces
            text = " ".join(text.split()).strip().split('.')[0] + '\n' # todo .
        else:
            text = text[0]["generated_text"].split('<|im_start|>assistant\n')[1].strip() + '\n'
    
    if text=="": # Error, the model didn't answer
        text='.' # This will prevent the evaluation script from failing
        
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generation')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('--peft_model_name_or_path', type=str, default="None")
    parser.add_argument('--quantization', action='store_true')
    parser.add_argument('--hugginface_token', type=str, default='')
    parser.add_argument('--test_filename', type=str, nargs='+', default=['data/test.jsonl'])
    parser.add_argument('--max_time', type=float, default=4.5)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--length_penalty', type=float, default=1.1)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument("--pretrained", action='store_true')
    args = parser.parse_args()

    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    for filename in args.test_filename:
        datasets = load_dataset('json', data_files=filename, split='train')
        outputs = generation(pipe, datasets, args)

        stem = pathlib.Path(filename).stem
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(f'{args.output_dir}/{stem}.txt', mode='w', encoding='utf-8') as f:
            f.writelines(outputs)
