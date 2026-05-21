import argparse
import re
import pandas as pd
from arguments.constants import TARGET, EXAMPLE, EXAMPLE_WITH_PROMPT, DEFINITION


def replace_prompt(x, pattern):
    match = re.search(pattern, x[EXAMPLE_WITH_PROMPT])
    return {EXAMPLE: x[EXAMPLE_WITH_PROMPT][: match.start()], TARGET: match.group(1)}


def clean_text(text):
    if isinstance(text, str):
        # Handle patterns like #{option1|option2}#
        text = re.sub(r"#\{[^|}]+\|([^}]+)\}#?", r"\1", text)
        # Handle patterns like {option1|option2}# at the start of the sentence
        text = re.sub(r"^\{[^|}]+\|([^}]+)\}#?", r"\1", text)
    return text


prompts = {
    "en": ". What is the definition of <TRG>?",
    "fr": ". Quelle est la définition de <TRG>?",
    "ru": ". Что такое <TRG>?",
    "no": ". Hva betyr <TRG>?",
    "de": ". Was ist die Definition von <TRG>?",
    "fi": ". Mitä tarkoittaa <TRG>?",
    "it": ". Qual è la definizione di <TRG>?",
    "es": ". ¿Cuál es la definición de <TRG>?",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, sep="\t")
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data[EXAMPLE] = data[EXAMPLE].apply(clean_text)
    data[EXAMPLE_WITH_PROMPT] = data.apply(
        lambda x: x[EXAMPLE].rstrip().rstrip(".").rstrip()
        + prompts[x["language"]].replace("<TRG>", x[TARGET].lower()),
        axis=1,
    )
    data[DEFINITION] = data[DEFINITION].apply(clean_text)
    data.to_csv(
        args.output_file,
        sep="\t",
        index=False,
        compression="gzip",
    )
