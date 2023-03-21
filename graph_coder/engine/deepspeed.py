#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch.distributed
from deepspeed.comm import init_distributed

from catalyst.engines import GPUEngine
from graph_coder.utils import print_rank0


class DeepspeedEngine(GPUEngine):
    """Force initialize torch nccl backend"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print_rank0(f"Torch is initialized: {torch.distributed.is_initialized()}")
        init_distributed("nccl", dist_init_required=True)