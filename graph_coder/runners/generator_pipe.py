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
from typing import Dict, Iterator, Union

import torch

from catalyst.utils import set_global_seed
from deepspeed import PipelineEngine
from deepspeed.runtime.dataloader import RepeatingLoader
from torch.utils.data import DataLoader, DistributedSampler

from graph_coder.runners.generator import GraphCoderGeneratorRunner


class GraphCoderGeneratorRunnerPipe(GraphCoderGeneratorRunner[PipelineEngine]):
    """Runner for graph-coder generator model"""

    model: PipelineEngine
    loaders: Dict[str, Union[DataLoader, Iterator]]

    def _run_loader(self) -> None:
        with torch.set_grad_enabled(self.is_train_loader):
            if self.is_train_loader:
                self.model.train_batch()
            else:
                self.model.eval_batch(data_iter=self.loader)

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
                self.loaders[key] = train_loader
                continue

            sampler = DistributedSampler(
                loader.dataset,
                num_replicas=self.model.dp_world_size,
                rank=self.model.mpu.get_data_parallel_rank(),
                shuffle=False,
            )
            # Build a loader and make it repeating.
            pipe_dataloader = self.model.deepspeed_io(
                loader.dataset, data_sampler=sampler
            )
            self.loaders[key] = RepeatingLoader(pipe_dataloader)
