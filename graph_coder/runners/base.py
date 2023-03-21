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
import abc
from typing import Dict, Generic, Optional, TypeVar

import torch
from accelerate import DistributedType

from catalyst import dl, metrics
from catalyst.core import IRunner
from torch import nn

from catalyst.utils import set_global_seed
from graph_coder.utils import summary

TM = TypeVar("TM", bound=nn.Module)


class GraphCoderRunnerBase(dl.Runner, abc.ABC, Generic[TM]):
    """Base class for graph-coder runners."""

    def __init__(
        self,
        model: TM,
        device: Optional[torch.device] = None,
        print_summary: bool = True,
        detect_anomaly: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if device is not None:
            self.model = model.to(device)
        else:
            self.model = model
        self.print_summary = print_summary
        self.detect_anomaly = detect_anomaly
        self.loss_metric = metrics.AdditiveMetric(compute_on_call=False)

    def on_experiment_start(self, runner: "IRunner"):
        super().on_experiment_start(runner)
        self._print_summary()
        torch.autograd.set_detect_anomaly(self.detect_anomaly)

    def on_loader_start(self, runner: "IRunner"):
        super().on_loader_start(runner)
        self.loss_metric.reset()

    def handle_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        loss = self._calc_loss(**batch)

        self.batch_metrics["loss"] = loss.item()
        self.loss_metric.update(loss.item(), self.batch_size)

        if self.is_train_loader:
            self.engine.backward(loss)
            if self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad()

    def on_loader_end(self, runner: "IRunner"):
        self.loader_metrics["loss"], _ = self.loss_metric.compute()
        super().on_loader_end(runner)

    def _print_summary(self):
        """Prints summary about the model"""
        if self.print_summary:
            sample_batch = next(iter(self.get_loaders()["train"]))
            summary(self.model, input_data=sample_batch)

    @abc.abstractmethod
    def _calc_loss(self, **kwargs: torch.Tensor) -> torch.Tensor:
        """Method that calculates loss for a batch."""

    def _setup_loaders(self) -> None:
        """Pass this to setup loader with deepspeed engine in `_setup_components`"""
        if self.engine.distributed_type != DistributedType.DEEPSPEED:
            super()._setup_loaders()

    def _setup_components(self) -> None:
        """Sets up components using deepspeed engine"""
        if self.engine.distributed_type != DistributedType.DEEPSPEED:
            super()._setup_loaders()
            return
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)
        self.model = self._setup_model()
        self.criterion = self._setup_criterion()

        (
            self.model,
            self.loaders["train"],
        ) = self.engine.prepare(
            self.model, self.loaders["train"]
        )
