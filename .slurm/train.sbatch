#! /bin/bash
#SBATCH --job-name="gc-train"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=2
#SBATCH --time=0-6:0
#SBATCH --mail-user=gaarutyunov@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=".slurm/.out/gc-train-"%j.out
#SBATCH --error=".slurm/.out/gc-train-"%j.err
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# modules
module restore default

# Executable
PL_RECONCILE_PROCESS=1 poetry run python graph_coder/train/train.py \
--log_every_n_steps=5 \
--batch-size=256 \
--accelerator=gpu \
--strategy=ddp_spawn
