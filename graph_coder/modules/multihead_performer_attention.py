#  Copyright (c) Microsoft Corporation.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
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

from typing import Optional

import torch
from einops import rearrange
from torch import nn, Tensor

from .multihead_attention import MultiheadAttention
from .performer.fast_attention import FastAttention


class MultiheadPerformerAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        performer_nb_features=None,
        performer_generalized_attention=False,
        attention_dropout=0.0,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        causal=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            attention_dropout,
            dropout,
            bias,
            self_attention,
            q_noise,
            qn_block_size,
        )
        assert attention_dropout == 0.0
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=causal,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=False,
        )

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def forward(
        self,
        query,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        assert attn_bias is None

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        assert k is not None
        assert k.size(0) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            key_padding_mask = key_padding_mask.to(torch.bool)[:, None, :, None]

        q, k, v = map(
            lambda t: rearrange(t, "n b (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )
        attn = self.fast_attention(q, k, v, key_padding_mask)
        attn = rearrange(attn, "b h n d -> n b (h d)")

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        return attn
