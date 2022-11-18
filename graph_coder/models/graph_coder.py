import math
from functools import reduce

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import LayerNorm
import torch.nn.functional as F

from graph_coder.modules.tokengt_graph_encoder import TokenGTGraphEncoder
from graph_coder.utils.activation import get_activation_fn


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class GraphCoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self._create_model()

    def _create_model(self):
        self.graph_encoder = TokenGTGraphEncoder(
            # <
            vocab_size=self.hparams.vocab_size,
            # >
            # < for embedding
            lap_node_id=self.hparams.lap_node_id,
            lap_node_id_k=self.hparams.lap_node_id_k,
            lap_node_id_sign_flip=self.hparams.lap_node_id_sign_flip,
            lap_node_id_eig_dropout=self.hparams.lap_node_id_eig_dropout,
            type_id=self.hparams.type_id,
            text_embed_size=self.hparams.text_embed_size,
            num_features=self.hparams.num_features,
            # >
            # <
            stochastic_depth=self.hparams.stochastic_depth,
            performer=self.hparams.performer,
            performer_finetune=self.hparams.performer_finetune,
            performer_nb_features=self.hparams.performer_nb_features,
            performer_feature_redraw_interval=self.hparams.performer_feature_redraw_interval,
            performer_generalized_attention=self.hparams.performer_generalized_attention,
            num_encoder_layers=self.hparams.encoder_layers,
            embedding_dim=self.hparams.encoder_embed_dim,
            ffn_embedding_dim=self.hparams.encoder_ffn_embed_dim,
            num_attention_heads=self.hparams.encoder_attention_heads,
            dropout=self.hparams.dropout,
            attention_dropout=self.hparams.attention_dropout,
            activation_dropout=self.hparams.act_dropout,
            encoder_normalize_before=self.hparams.encoder_normalize_before,
            layernorm_style=self.hparams.layernorm_style,
            activation_fn=self.hparams.activation_fn,
            return_attention=self.hparams.return_attention,
            repr_mode=self.hparams.repr_mode
            # >
        )

        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = self.hparams.remove_head
        self.masked_lm_pooler = nn.Linear(
            self.hparams.encoder_embed_dim, self.hparams.encoder_embed_dim
        )
        self.lm_head_transform_weight = nn.Linear(
            self.hparams.encoder_embed_dim, self.hparams.encoder_embed_dim
        )
        self.activation_fn = get_activation_fn(self.hparams.activation_fn)
        self.layer_norm = LayerNorm(self.hparams.encoder_embed_dim)

        self.embed_out = nn.Linear(
            self.hparams.encoder_embed_dim,
            self.hparams.vocab_size,
            bias=False,
        )

        if self.hparams.performer_finetune:
            self.performer_finetune_setup()
        self._reset_parameters()

    def _reset_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None):
        inner_states, attn_dict = self.graph_encoder(batched_data, perturb=perturb)

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        # project masked tokens only
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.hparams.share_encoder_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        if self.hparams.return_attention:
            return x, attn_dict
        else:
            return x

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

    def _calculate_loss(self, batch, mode="train"):
        (
            padded_index,
            padded_feature,
            padding_mask,
            padded_node_mask,
            padded_edge_mask,
        ) = self.graph_encoder.graph_feature.get_batch(
            batch["node_data"],
            batch["edge_index"],
            batch["edge_data"],
            batch["node_num"],
            batch["edge_num"],
        )
        labels = torch.zeros(
            (padded_feature.size(0), padded_feature.size(1), padded_feature.size(2), self.hparams.vocab_size),
            dtype=torch.int64,
        )
        labels.scatter_(-1, padded_feature, 1)
        preds = self.forward(batch)
        loss = F.nll_loss(
            preds.view(-1),
            labels.view(-1),
            ignore_index=0
        )

        self.log("%s_loss" % mode, loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
