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

from collections import defaultdict
from typing import Mapping, Any

import torch
from catalyst import dl
from torch import nn

from graph_coder.data import GraphCoderBatch


class GraphCoderGeneratorRunner(dl.Runner):
    def __init__(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_batch(self, batch: GraphCoderBatch, **kwargs) -> Mapping[str, Any]:
        pass

    def on_batch_start(self, runner: "IRunner"):
        # noinspection PyTypeChecker
        batch: GraphCoderBatch = self.batch
        self.batch_size = batch.idx.size(0)
        self.batch_step += self.engine.num_processes
        self.loader_batch_step += self.engine.num_processes
        self.sample_step += self.batch_size * self.engine.num_processes
        self.loader_sample_step += self.batch_size * self.engine.num_processes
        self.batch_metrics = defaultdict(None)

    def handle_batch(self, batch: GraphCoderBatch) -> None:
        loss = self._calc_loss(batch)

        self.batch_metrics.update({"loss", loss.item()})

        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _calc_loss(self, batch: GraphCoderBatch) -> torch.Tensor:
        result = self.model(batch)

        lm_logits = []
        target_ids = []

        if batch.has_docstring:
            target_ids.append(batch.docstring[batch.docstring_attn_mask.bool()])
            lm_logits.append(result["docstring"][batch.docstring_attn_mask.bool()])
        if batch.has_graph:
            if len(target_ids) != 0:
                device = batch.node_data.device
                target_ids.append(
                    torch.tensor([self.model.eos_token_id], device=device)
                )
                lm_logits.append(
                    torch.tensor([self.model.eos_token_id], device=device).repeat(
                        1, self.model.vocab_size
                    )
                )

            target_ids.extend(
                [
                    batch.node_data[batch.node_data_attn_mask.bool()],
                    batch.edge_data[batch.edge_data_attn_mask.bool()],
                ]
            )
            masks = torch.cat(
                [
                    batch.node_data_attn_mask.view(-1),
                    batch.edge_data_attn_mask.view(-1),
                ]
            ).bool()
            lm_logits.append(result["graph"].view(-1, self.model.vocab_size)[masks, :])
        if batch.has_source:
            if len(target_ids) != 0:
                device = batch.source.device
                target_ids.append(
                    torch.tensor([self.model.eos_token_id], device=device)
                )
                lm_logits.append(
                    torch.tensor([self.model.eos_token_id], device=device).repeat(
                        1, self.model.vocab_size
                    )
                )

            target_ids.append(batch.source[batch.source_attn_mask.bool()])
            lm_logits.append(result["source"][batch.source_attn_mask.bool()])

        target_ids = torch.cat(target_ids)
        lm_logits = torch.cat(lm_logits, dim=0)

        shift_logits = lm_logits[:-1, :].contiguous()
        shift_labels = target_ids[1:].contiguous().long()

        return self.criterion(shift_logits, shift_labels.long())
