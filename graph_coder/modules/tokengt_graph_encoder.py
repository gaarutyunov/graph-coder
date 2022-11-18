"""
Modified from https://github.com/jw9730/tokengt
"""

import torch
import torch.nn as nn

from performer_pytorch import ProjectionUpdater
from torch.nn import Dropout, LayerNorm

from .tokenizer import GraphFeatureTokenizer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer


class TokenGTGraphEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        text_embed_size: int,
        num_features: int,
        lap_node_id: bool = False,
        lap_node_id_k: int = 8,
        lap_node_id_sign_flip: bool = False,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = False,
        stochastic_depth: bool = False,
        performer: bool = False,
        performer_finetune: bool = False,
        performer_nb_features: int = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        performer_auto_check_redraw: bool = True,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "postnorm",
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        return_attention: bool = False,
        repr_mode: str = "embedding",
    ) -> None:

        super().__init__()
        self.dropout_module = Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.performer = performer
        self.performer_finetune = performer_finetune

        self.graph_feature = GraphFeatureTokenizer(
            vocab_size=vocab_size,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            repr_mode=repr_mode,
            text_embed_size=text_embed_size,
            num_features=num_features,
        )
        self.performer_finetune = performer_finetune
        self.embed_scale = embed_scale

        self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if layernorm_style == "prenorm":
            self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([])

        if stochastic_depth:
            assert layernorm_style == "prenorm"  # only for residual nets

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                performer_nb_features,
                performer_generalized_attention,
                performer_auto_check_redraw,
                performer_feature_redraw_interval,
            )
            self.performer = False
            performer = False
            performer_nb_features = None
            performer_generalized_attention = False
            performer_auto_check_redraw = False
            performer_feature_redraw_interval = None

        self.layers.extend(
            [
                self.build_tokengt_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    encoder_layers=num_encoder_layers,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=(0.1 * (layer_idx + 1) / num_encoder_layers)
                    if stochastic_depth
                    else 0,
                    performer=performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    activation_fn=activation_fn,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        if performer:
            # keeping track of when to redraw projections for all attention layers
            self.performer_auto_check_redraw = performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(
                self.layers, performer_feature_redraw_interval
            )

    def performer_fix_projection_matrices_(self):
        self.performer_proj_updater.feature_redraw_interval = None

    def performer_finetune_setup(self):
        assert self.performer_finetune
        (
            performer_nb_features,
            performer_generalized_attention,
            performer_auto_check_redraw,
            performer_feature_redraw_interval,
        ) = self.cached_performer_options

        for layer in self.layers:
            layer.performer_finetune_setup(
                performer_nb_features, performer_generalized_attention
            )

        self.performer = True
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.performer_proj_updater = ProjectionUpdater(
            self.layers, performer_feature_redraw_interval
        )

    def build_tokengt_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        encoder_layers,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        drop_path,
        performer,
        performer_nb_features,
        performer_generalized_attention,
        activation_fn,
        layernorm_style,
        return_attention,
    ):
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            layernorm_style=layernorm_style,
            return_attention=return_attention,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = True,
    ):
        if self.performer and self.performer_auto_check_redraw:
            self.performer_proj_updater.redraw_projections()

        x, padding_mask, padded_index = self.graph_feature(batched_data, perturb)

        # x: B x T x F x D

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x F x D -> T x B x F x D
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        attn_dict = {"maps": {}, "padded_index": padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
            )
            if not last_state_only:
                inner_states.append(x)
            attn_dict["maps"][i] = attn

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), attn_dict
        else:
            return inner_states, attn_dict
