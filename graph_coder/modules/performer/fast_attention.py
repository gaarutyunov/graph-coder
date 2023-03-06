#  MIT License
#
#  Copyright (c) 2020 Phil Wang
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from functools import partial

import torch
from performer_pytorch.performer_pytorch import (
    generalized_kernel,
    softmax_kernel,
    linear_attention,
    FastAttention as PerformerFastAttention,
)


class FastAttention(PerformerFastAttention):
    def forward(self, q, k, v, key_padding_mask=None) -> torch.Tensor:
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device,
            )
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn

        if key_padding_mask is not None:
            k = k.masked_fill(key_padding_mask, 0.0)
            v = v.masked_fill(key_padding_mask, 0.0)

        out = attn_fn(q, k, v)

        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask, 0.0)

        return out
