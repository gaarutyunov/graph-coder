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

# Load modules and activate virtual environment

function separator() {
    printf '\n'
    printf '=%.0s' {1..80}
    printf '\n'
    printf '\n'
}

function print() {
    printf '%s\n\n' "$1"
}

source ~/.bashrc || print 'No ~/.bashrc'
module restore default || print 'Not running in slurm environment'
source activate graph-coder || print 'No env graph-coder'

print 'Loaded modules:'
module list || print 'Not running in slurm environment'

separator

print 'Python path:'
type python

separator

print 'Conda environment:'
conda list || print 'No conda available'

separator

print 'Cuda version:'
nvcc --version || print 'No cuda available'

separator

print 'PyTorch environment:'
python -m torch.utils.collect_env

separator

print 'DeepSpeed report:'
ds_report || print 'No deepspeed available'

separator