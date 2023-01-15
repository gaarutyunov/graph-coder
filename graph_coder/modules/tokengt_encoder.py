import torch
from fairseq import utils
from torch import nn
from torch.nn import LayerNorm
from torch.functional import F

from graph_coder.modules.tokengt_graph_encoder import TokenGTGraphEncoder


class TokenGTEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        max_nodes: int = 1000,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 8,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 768,
        act_dropout: float = 0.1,
        share_encoder_input_output_embed: bool = None,
        remove_head: bool = False,
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
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "prenorm",
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        return_attention: bool = False,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.encoder_layers = encoder_layers
        self.num_attention_heads = encoder_attention_heads
        self.return_attention = return_attention

        if layernorm_style not in ["prenorm", "postnorm"]:
            raise NotImplementedError

        self.graph_encoder = TokenGTGraphEncoder(
            # < for tokenization
            embedding=embedding,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            # >
            stochastic_depth=stochastic_depth,
            performer=performer,
            performer_finetune=performer_finetune,
            performer_nb_features=performer_nb_features,
            performer_feature_redraw_interval=performer_feature_redraw_interval,
            performer_generalized_attention=performer_generalized_attention,
            num_encoder_layers=encoder_layers,
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_ffn_embed_dim,
            num_attention_heads=encoder_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=act_dropout,
            encoder_normalize_before=encoder_normalize_before,
            layernorm_style=layernorm_style,
            apply_graphormer_init=apply_graphormer_init,
            activation_fn=activation_fn,
            return_attention=return_attention,
        )

        self.share_input_output_embed = share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not remove_head
        self.masked_lm_pooler = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.lm_head_transform_weight = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(encoder_embed_dim)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep, attn_dict = self.graph_encoder(
            batched_data, perturb=perturb
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        if self.return_attention:
            return x, attn_dict
        else:
            return x

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

