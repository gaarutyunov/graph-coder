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
from typing import Any, Generic, Mapping, TypeVar

import torch
from torch import nn

from graph_coder.runners.base import GraphCoderRunnerBase


TMM = TypeVar("TMM", bound=nn.Module)


class GraphCoderGeneratorRunner(GraphCoderRunnerBase[TMM], Generic[TMM]):
    """Runner for graph-coder generator model"""

    def __init__(
        self,
        model: TMM,
        eos_token_id: int,
        vocab_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        return self.model.generate(**batch)  # type: ignore[operator]

    def _calc_loss(self, **kwargs: torch.Tensor) -> torch.Tensor:
        """Calculate loss for the given batch"""
        shift_logits = self.model(**kwargs)
        shift_labels = kwargs["labels"]

        return self.criterion(shift_logits, shift_labels)
