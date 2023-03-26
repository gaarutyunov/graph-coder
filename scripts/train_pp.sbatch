#! /bin/bash
#
# Copyright 2023 German Arutyunov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#SBATCH --mail-user=germanarutyunov@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=gc-train
#SBATCH --output="logs/slurm/train/"%j.out
#SBATCH --error="logs/slurm/train/"%j.err
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=7-0:0

# Load modules and activate virtual environment
chmod +x ./scripts/prepare.sh
source ./scripts/prepare.sh

# Train model
#NCCL_DEBUG=info NCCL_DEBUG_SUBSYS=GRAPH ACCELERATE_LOG_LEVEL=DEBUG deepspeed --module --no_local_rank graph_coder.run --spawn --root "$1" "${@:2}"
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL ACCELERATE_LOG_LEVEL=DEBUG torchrun --standalone --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID --module graph_coder.run --spawn --root "$1" "${@:2}"
