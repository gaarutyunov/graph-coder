#! /bin/bash
#SBATCH --job-name="gc-data"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --ntasks=1
#SBATCH --time=0-6:0
#SBATCH --mail-user=gaarutyunov@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=".slurm/out/gc-data"%j.out
#SBATCH --error=".slurm/out/gc-data"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_d"

# modules
module restore default

# Executable
poe dataset_process
poe tokenizer_train
poe tokenizer_transform
