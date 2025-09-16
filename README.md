# MultilingualDefGen

#### Download and process axolotl24_shared_task data
```bash bash/Axolotl24.sh```

#### Download and process dbnary data
```bash bash/Dbnary.sh```

#### Train-dev-test split
```bash bash/train_dev_test.sh```

#### Fine-tuning
```bash bash/finetuning.sh```

### Generation / Evaluation Dev
```bash bash/generation_dev.sh```
```bash bash/evaluation_dev.sh```

### Generation / Evaluation Test
```bash bash/generation_test.sh```
```bash bash/evaluation_test.sh```

```TowerInstruct and Dictionary.zip``` includes:
- the ```predictions``` folder: it contains the definitions (line-by-line text files) generated in our work by the TowerInstruct and Dictionary models;
- the ```submissions``` folder: it contains definitions in the Axoltl prediction format;
- the ```evaluation``` folder: it contains evaluation results produced by the Axoltl evaluation script for Subtask 2.
