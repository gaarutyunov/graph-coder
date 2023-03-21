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
from collections import defaultdict
from typing import Dict, Iterator, Union

import torch

from catalyst.core import IRunner, IRunnerError
from catalyst.core.misc import get_loader_num_samples
from catalyst.utils import maybe_recursive_call, set_global_seed
from deepspeed import PipelineEngine
from deepspeed.runtime.data_pipeline.data_sampling.data_sampler import (
    DeepSpeedDataSampler,
)
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from graph_coder.pipe.dataloader import PipeLoaderWrapper
from graph_coder.runners.generator import GraphCoderGeneratorRunner


class GraphCoderGeneratorRunnerPipe(GraphCoderGeneratorRunner[PipelineEngine]):
    """Runner for graph-coder generator model"""

    model: PipelineEngine
    loader: Union[DeepSpeedDataLoader, DataLoader]
    loaders: Dict[str, Union[DataLoader, Iterator]]

    def on_loader_start(self, runner: IRunner):
        """Event handler."""
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        assert self.is_train_loader or self.is_valid_loader or self.is_infer_loader
        self.loader_batch_size: int = self.loader.batch_size
        self.loader_batch_len: int = len(self.loader)
        if isinstance(self.loader.data_sampler, (DistributedSampler, RandomSampler)):
            self.loader_sample_len: int = self.loader.data_sampler.num_samples
        elif isinstance(self.loader.data_sampler, DeepSpeedDataSampler):
            self.loader_sample_len: int = self.loader.data_sampler.total_samples
        else:
            self.loader_sample_len: int = get_loader_num_samples(self.loader)
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        if self.loader_batch_len == 0:
            raise IRunnerError(f"DataLoader with name {self.loader_key} is empty.")
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)

        maybe_recursive_call(self.model, "train", mode=self.is_train_loader)
        if isinstance(self.loader.data_sampler, DistributedSampler):
            self.loader.data_sampler.set_epoch(self.epoch_step)
        self.loss_metric.reset()

    def _run_loader(self) -> None:
        with torch.set_grad_enabled(self.is_train_loader):
            if self.is_train_loader:
                self.model.train_batch(data_iter=PipeLoaderWrapper(self.loader))
            else:
                self.model.eval_batch(data_iter=PipeLoaderWrapper(self.loader))

    def _setup_loaders(self) -> None:
        """Pass this to setup loader with deepspeed engine in `_setup_components`"""

    def _setup_components(self) -> None:
        """Sets up components using deepspeed engine"""
        set_global_seed(self.seed + max(0, self.engine.process_index) + self.epoch_step)
        self.model = self._setup_model()  # type: ignore[assignment]
        self.criterion = self._setup_criterion()
        self.loaders = {}

        (
            self.model,
            train_loader,
        ) = self.engine.prepare(self.model, self.get_loaders()["train"])

        loaders = self.get_loaders()

        for key, loader in loaders.items():
            if key == "train":
                self.loaders[key] = train_loader.loader
                continue

            sampler = DistributedSampler(
                loader.dataset,
                num_replicas=self.model.dp_world_size,
                rank=self.model.mpu.get_data_parallel_rank(),
                shuffle=False,
            )
            # Build a loader and make it repeating.
            self.loaders[key] = self.model.deepspeed_io(
                loader.dataset,
                data_sampler=sampler,
            )
