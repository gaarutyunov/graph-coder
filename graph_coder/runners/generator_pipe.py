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

import torch

from graph_coder.runners.generator import GraphCoderGeneratorRunner


class GraphCoderGeneratorRunnerPipe(GraphCoderGeneratorRunner):
    """Runner for graph-coder generator model"""
    def _run_loader(self) -> None:
        if self.is_train_loader:
            with torch.set_grad_enabled(self.is_train_loader):
                self.model.train_batch(data_iter=iter(self.loader))
        else:
            raise NotImplementedError("only training is implemented for pipeline parallel model")
