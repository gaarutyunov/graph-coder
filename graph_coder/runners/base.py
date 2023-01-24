#  Copyright 2023 German Arutyunov
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
import abc
from collections import defaultdict
from typing import Dict

import torch
from catalyst import dl
from catalyst.core import IRunner
from torch import nn

from graph_coder.data import GraphCoderBatch
from graph_coder.utils import summary


class GraphCoderRunnerBase(dl.Runner, abc.ABC):
    """Base class for graph-coder runners."""

    def __init__(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model

    def on_batch_start(self, runner: IRunner) -> None:
        # noinspection PyTypeChecker
        batch: GraphCoderBatch = self.batch  # type: ignore
        self.batch_size = batch.batch_size
        self.batch_step += self.engine.num_processes
        self.loader_batch_step += self.engine.num_processes
        self.sample_step += self.batch_size * self.engine.num_processes
        self.loader_sample_step += self.batch_size * self.engine.num_processes
        self.batch_metrics: Dict[str, float] = defaultdict(None)

    def on_experiment_start(self, runner: "IRunner"):
        super().on_experiment_start(runner)
        self._print_summary()

    def handle_batch(self, batch: GraphCoderBatch) -> None:
        loss = self._calc_loss(batch)

        self.batch_metrics["loss"] = loss.item()

        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _print_summary(self):
        """Prints summary about the model"""
        sample_batch = next(iter(self.get_loaders()["train"]))
        summary(self.model, input_data=sample_batch)

    @abc.abstractmethod
    def _calc_loss(self, batch: GraphCoderBatch) -> torch.Tensor:
        """Method that calculates loss for a batch."""
        pass
