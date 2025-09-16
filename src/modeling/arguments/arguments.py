# inspired by https://towardsdatascience.com/an-entry-point-into-huggingface-2f3d1e60ad5a/
from typing import Optional
from dataclasses import dataclass, field

from arguments.constants import EXAMPLE_WITH_PROMPT, DEFINITION

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default='Unbabel/TowerInstruct-7B-v0.2',
    )
    tag: str = field(
        metadata={"help": "tag to be appended to the finetuned model name"},
        default="defmod"
    )
    cache_dir: str = field(default='~/.cache/huggingface/')

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_filename: str = field()
    dev_filename: str = field()
    context_column: str = field(default=EXAMPLE_WITH_PROMPT)
    gloss_column: str = field(default=DEFINITION)
    max_target_length: int = field(default=122)

@dataclass
class PEFTArguments:
    lora_rank: int = field(default=256)
    lora_alpha: int = field(default=512)
    lora_dropout: float = field(default=0.1)
    max_seq_length: int = field(default=192)
