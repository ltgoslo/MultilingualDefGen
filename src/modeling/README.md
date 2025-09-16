# Saga

## Fine-tune decoder-only

```
sbatch decoder_only_finetuning_saga.slurm --output_dir <output dir> \
                         --weight_decay 0.001 \
                         --warmup_ratio 0.05 \
                         --per_device_train_batch_size 16 \
                         --per_device_eval_batch_size 16 \
                         --learning_rate 1e-4 \
                         --cache_dir /cluster/projects/nn9851k/models/ \
                         --train_filename /cluster/projects/nn9851k/corpora/defgen_data/train_axolotl24st_ru.tsv.gz \
                         --dev_filename /cluster/projects/nn9851k/corpora/defgen_data/dev_axolotl24st_ru.tsv \
                         --qlora \
                         --lora \
                         --optim paged_adamw_8bit -\
                         --tag adamw8bit_325 \
                          --num_train_epochs 1 \
                          --context_column example \
                          --max_seq_length 325
```

## Predict decoder-only

see lumi section and change paths and script name to saga
     
# Lumi

## Fine-tune

```
sbatch encoder_decoder_lumi.slurm    --train_filename /scratch/project_465001925/corpora/defgen_data/train_axolotl24st_ru.tsv.gz \
    --dev_filename /scratch/project_465001925/corpora/defgen_data/dev_axolotl24st_ru.tsv \
    --output_dir /scratch/project_465001925/models/DefGen/aya-qlora-all-linear-new-314 \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 1 \
    --model_name_or_path CohereLabs/aya-101 \
    --qlora \
    --context_column example \
    --optim paged_adamw_8bit \
    --cache_dir /scratch/project_465001925/cache/ \
    --max_seq_length 314
```

## Predict

### Encoder-only
```
sbatch encoder_decoder_predict_lumi.slurm     --model /scratch/project_465001925/models/DefGen/aya-qlora-all-linear-new-314/CohereLabs-aya-101-defmod/final-epoch/     --test_filename /scratch/project_465001925/corpora/defgen_data/dev_axolotl24st_ru.tsv.gz     --bsize 48     --sampling 1     --rpenalty 1.1     --cache_dir /scratch/project_465001925/cache/ --filter 0 \
    --num_beams 5
```

### Decoder-only

```
sbatch decoder_only_predict_lumi.slurm --quantization \
    --batch_size 16 \
    --peft_model_name_or_path /scratch/project_465001925/models/DefGen/Unbabel-TowerInstruct-7B-v0.2-adamw8bit_325train_axolotl24st_ru/final-epoch/ \
    --test_filename /scratch/project_465001925/corpora/defgen_data/dev_axolotl24st_ru.tsv.gz \
    --cache_dir /scratch/project_465001925/cache/ \
    --max_new_tokens 60 \
    --strategy beam_search_multinomial_sampling
```
