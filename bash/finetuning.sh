sbatch --time 3-10:00:00 --job-name de_D_finetuning --output=slurm_output/finetuning/de.out bash/_finetuning.sh de Dbnary
sbatch --time 01-20:00:00 --job-name ru_D+A_finetuning --output=slurm_output/finetuning/ru.out bash/_finetuning.sh ru Dbnary+Axoltl
sbatch --time 01-20:00:00 --job-name fi_D+A_finetuning --output=slurm_output/finetuning/fi.out bash/_finetuning.sh fi Dbnary+Axoltl
sbatch --time 01-20:00:00 --job-name ru_A_finetuning --output=slurm_output/finetuning/ru.out bash/_finetuning.sh ru Axoltl
sbatch --time 01-20:00:00 --job-name fi_A_finetuning --output=slurm_output/finetuning/fi.out bash/_finetuning.sh fi Axoltl
