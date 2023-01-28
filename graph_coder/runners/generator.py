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
from pathlib import Path
from typing import Mapping, Any

import torch
from torch import nn

from graph_coder.data import GraphCoderBatch
from graph_coder.runners.base import GraphCoderRunnerBase


class GraphCoderGeneratorRunner(GraphCoderRunnerBase):
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

    def _calc_loss(self, batch: GraphCoderBatch) -> torch.Tensor:
        """Calculate loss for the given batch"""
        result = self.model(batch)

        lm_logits = []
        target_ids = []

        if batch.has_docstring:
            target_ids.append(batch.docstring[batch.docstring_attn_mask.bool()])
            lm_logits.append(result["docstring"][batch.docstring_attn_mask.bool()])
            # add eos token
            device = batch.docstring.device
            target_ids.append(torch.tensor([self.eos_token_id], device=device))
            lm_logits.append(
                torch.tensor([self.eos_token_id], device=device).repeat(
                    1, self.vocab_size
                )
            )
        if batch.has_graph:
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
                ],
                dim=1,
            ).bool()
            try:
                lm_logits.append(
                    result["graph"].view(-1, self.vocab_size)[masks, :]
                )  # TODO: check this for [23496, 38958, 10398, 14494]
            except Exception as e:
                debug_path = Path("debug")
                debug_path.mkdir(exist_ok=True)
                log_txt = debug_path / "log.txt"
                with open(log_txt, "w") as f:
                    print(f"Masks shape: {masks.shape}\n", file=f)
                    print(f'Graph shape: {result["graph"].shape}\n', file=f)
                    print(f"Node data shape: {batch.node_data.shape}\n", file=f)
                    print(f"Edge data shape: {batch.edge_data.shape}\n", file=f)
                    print(
                        f"Node data attn mask shape: {batch.node_data_attn_mask.shape}\n",
                        file=f,
                    )
                    print(
                        f"Edge data attn mask shape: {batch.edge_data_attn_mask.shape}\n",
                        file=f,
                    )
                    print(f"Node data device: {batch.node_data.device}\n", file=f)
                    print(f"Edge data device: {batch.edge_data.device}\n", file=f)
                    print(f"Masks device: {masks.device}\n", file=f)
                    print(f'Graph device: {result["graph"].device}\n', file=f)
                torch.save(batch.node_data.cpu(), debug_path / "node_data.pt")
                torch.save(batch.edge_data.cpu(), debug_path / "edge_data.pt")
                torch.save(
                    batch.node_data_attn_mask.cpu(),
                    debug_path / "node_data_attn_mask.pt",
                )
                torch.save(
                    batch.edge_data_attn_mask.cpu(),
                    debug_path / "edge_data_attn_mask.pt",
                )
                torch.save(masks.cpu(), debug_path / "masks.pt")
                torch.save(result["graph"].cpu(), debug_path / "graph.pt")
                raise Exception(
                    f"Error in loss calculation for {batch.idx.tolist()}"
                ) from e
            # add eos token
            device = batch.node_data.device
            target_ids.append(torch.tensor([self.eos_token_id], device=device))
            lm_logits.append(
                torch.tensor([self.eos_token_id], device=device).repeat(
                    1, self.vocab_size
                )
            )
        if batch.has_source:
            target_ids.append(batch.source[batch.source_attn_mask.bool()])
            lm_logits.append(result["source"][batch.source_attn_mask.bool()])
            # add eos token
            device = batch.source.device
            target_ids.append(torch.tensor([self.eos_token_id], device=device))
            lm_logits.append(
                torch.tensor([self.eos_token_id], device=device).repeat(
                    1, self.vocab_size
                )
            )

        target_ids_ = torch.cat(target_ids)
        lm_logits_ = torch.cat(lm_logits, dim=0)

        shift_logits = lm_logits_[:-1, :].contiguous()
        shift_labels = target_ids_[1:].contiguous().long()

        return self.criterion(shift_logits, shift_labels.long())
