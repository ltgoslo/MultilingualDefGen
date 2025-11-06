# Explaining novel senses using definition generation with open language models

This repository contains the code accompanying the paper [Explaining novel senses using definition generation with open language models](https://aclanthology.org/2025.findings-emnlp.1214/) by Mariia Fedorova, Andrey Kutuzov, Francesco Periti, and Yves Scherrer, published at EMNLP'25 Findings.

We apply definition generators based on open-weights large language models to the task of creating explanations of novel senses, taking target word usages as an input. To this end, we employ the datasets from the [AXOLOTL’24 shared task](https://github.com/ltgoslo/axolotl24_shared_task) on explainable semantic change modeling, which features Finnish, Russian and German languages.
We fine-tune and [provide publicly open-source definition generation  models](https://huggingface.co/collections/ltg/definition-modeling) performing better than the best submissions of the aforementioned shared task, which employed closed proprietary LLMs. In addition, we find that encoder-decoder definition generators perform on par with their decoder-only counterparts.

## Download and process axolotl24_shared_task data
```bash bash/Axolotl24.sh```

## Download and process dbnary data
```bash bash/Dbnary.sh```

## Train-dev-test split
```bash bash/train_dev_test.sh```

## Fine-tuning and Generation
 
Instructions for running fine-tuning and prediction are to be found in `src/modeling`.

## Evaluation 

`src/evaluate.sh` runs mapping predicted definitions to the shared task input files, selecting a single definition for senses with many usage examples and scoring with the shared task evaluation script.

```commandline
cd src/
./evaluate.sh <file with predicted definitions> <file with gold definitions>
```
It produces many files from different processing stages. Metrics' values will be in *.scores_sample.csv

t-test may be run as 

```
cd src/
python3 t_test --a <*.scores_sample.csv from the 1st model> --b <*.scores_sample.csv from the 2nd model>
```

## Citing us

```bibtex
@inproceedings{fedorova-etal-2025-explaining,
    title = "Explaining novel senses using definition generation with open language models",
    author = "Fedorova, Mariia  and
      Kutuzov, Andrey  and
      Periti, Francesco  and
      Scherrer, Yves",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1214/",
    pages = "22294--22302",
    ISBN = "979-8-89176-335-7",
    abstract = "We apply definition generators based on open-weights large language models to the task of creating explanations of novel senses, taking target word usages as an input. To this end, we employ the datasets from the AXOLOTL{'}24 shared task on explainable semantic change modeling, which features Finnish, Russian and German languages. We fine-tune and provide publicly the open-source models performing higher than the best submissions of the aforementioned shared task, which employed closed proprietary LLMs. In addition, we find that encoder-decoder definition generators perform on par with their decoder-only counterparts."
}
```
