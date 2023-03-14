#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Any, Mapping

import torch
from torch import nn

from graph_coder.data import GraphCoderBatch
from graph_coder.runners.generator import GraphCoderGeneratorRunner


class GraphCoderGeneratorRunnerPipe(GraphCoderGeneratorRunner):
    """Runner for graph-coder generator model"""

    def __init__(
        self,
        model: nn.Module,
        eos_token_id: int,
        vocab_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def predict_batch(self, batch: GraphCoderBatch, **kwargs) -> Mapping[str, Any]:
        # TODO: split prediction into graph and source
        return {"predictions": self.model(batch, **kwargs)}

    def _calc_loss(self, **kwargs: torch.Tensor) -> torch.Tensor:
        """Calculate loss for the given batch"""
        return self.forward_model(**kwargs)
