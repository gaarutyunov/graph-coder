#! /bin/bash

#
# Copyright 2023 German Arutyunov
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

# Load modules and activate virtual environment
source ~/.bashrc
module restore default
source activate graph-coder

function separator() {
    printf '\n'
    printf '=%.0s' {1..80}
}

printf 'Loaded modules:\n\n'
module list

separator

printf 'Python path:\n\n'
type python

separator

printf 'Conda environment graph-coder:\n\n'
conda list

separator

printf 'Cuda version:\n\n'
nvcc --version

separator

printf 'PyTorch environment:\n\n'
python -m torch.utils.collect_env

separator