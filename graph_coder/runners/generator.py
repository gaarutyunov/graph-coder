from typing import Mapping, Any

import torch
from catalyst import dl
from torch import nn

from graph_coder.data.collator import GraphCoderBatch


class GraphCoderGeneratorRunner(dl.Runner):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = criterion

    def predict_batch(self, batch: GraphCoderBatch, **kwargs) -> Mapping[str, Any]:
        pass

    def handle_batch(self, batch: GraphCoderBatch) -> None:
        lm_logits = self.model(batch)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        target_ids = torch.Tensor()  # TODO: define target_ids
        shift_labels = target_ids[..., 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        self.batch_metrics.update({"loss", loss})

        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
