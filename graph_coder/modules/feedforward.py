"""
Modified from https://github.com/jw9730/tokengt
"""

import torch.nn as nn
from torch.nn import Dropout

from graph_coder.utils.activation import get_activation_fn


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        activation_fn,
        activation_dropout,
        dropout,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.activation_dropout_module = Dropout(activation_dropout)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.dropout_module = Dropout(dropout)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
