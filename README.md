# MultilingualDefGen

#### Download and process axolotl24_shared_task data
```bash bash/Axolotl24.sh```

#### Download and process dbnary data
```bash bash/Dbnary.sh```

#### Train-dev-test split
```bash bash/train_dev_test.sh```

#### Fine-tuning and Generation
 
Instructions for running fine-tuning and prediction are to be found in `src/modeling`.

#### Evaluation 

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

Cite us

```bibtex
@misc{fedorova2025explainingnovelsensesusing,
      title={Explaining novel senses using definition generation with open language models}, 
      author={Mariia Fedorova and Andrey Kutuzov and Francesco Periti and Yves Scherrer},
      year={2025},
      eprint={2509.26181},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26181}, 
}
```