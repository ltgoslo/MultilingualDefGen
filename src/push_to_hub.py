import torch
import argparse
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(args, tokenizer):
    login(args.read_hugginface_token)
    settings = dict(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    device_map='auto')

    settings['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(**settings)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    peft_model = PeftModel.from_pretrained(model, args.peft_model_name_or_path)
    peft_model.eval()
    return model


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              padding_side="left",
                                              add_eos_token=True,
                                              add_bos_token=True)
    tokenizer.add_tokens(["<stop>"])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def push_to_hub(args):
    login(args.read_hugginface_token)

    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer)

    login(args.write_hugginface_token)
    model.push_to_hub(args.model_id, private=True)
    tokenizer.push_to_hub(args.model_id, private=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_hugginface_token", type=str)
    parser.add_argument("--write_hugginface_token", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Unbabel/TowerInstruct-7B-v0.2")
    parser.add_argument("--peft_model_name_or_path", type=str)
    parser.add_argument("--model_id", type=str)
    args = parser.parse_args()

    push_to_hub(args)
