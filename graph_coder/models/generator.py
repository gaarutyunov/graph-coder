import torch
from torch import nn

from graph_coder.data.collator import GraphCoderBatch


class GraphCoderGenerator(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        encoder: nn.Module,
        graph_encoder: nn.Module,
        decoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id
        self.encoder = encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, batch: GraphCoderBatch) -> torch.Tensor:
        x = []
        tgt = []
        if batch.docstring is not None and batch.docstring.size(-1) > 0:
            emb = self.embedding(batch.docstring)
            docstring_encoded = self.encoder(emb)
            x.append(docstring_encoded)
            tgt.append(emb)
        if batch.node_data is not None and batch.node_data.size(-2) > 0:
            graph_encoded = self.graph_encoder(batch)
            if len(x) != 0:
                eos = torch.Tensor([self.eos_token_id]).repeat(graph_encoded.size(0), 1, graph_encoded.size(-1))
                x.append(eos)
                tgt.append(eos)
            x.append(graph_encoded)
            tgt.append(self.embedding(torch.cat([
                batch.node_data,
                batch.edge_data,
            ], dim=1)).sum(-2))
        if batch.source is not None and batch.source.size(-1) > 0:
            emb = self.embedding(batch.source)
            source_code_encoded = self.encoder(emb)
            if len(x) != 0:
                eos = torch.Tensor([self.eos_token_id]).repeat(source_code_encoded.size(0), 1, source_code_encoded.size(-1))
                x.append(eos)
                tgt.append(eos)
            x.append(source_code_encoded)
            tgt.append(emb)
        x = torch.cat(x, dim=1)
        tgt = torch.cat(tgt, dim=1)

        out = self.decoder(tgt, x)
        hidden_states = torch.tanh(self.dense(out)).contiguous()
        lm_logits = self.lm_head(hidden_states)

        return lm_logits
