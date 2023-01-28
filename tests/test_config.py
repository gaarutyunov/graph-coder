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

import torch
from catalyst.contrib.scripts.run import process_configs
from catalyst.registry import REGISTRY
from torch.nn import TransformerDecoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graph_coder.config import ConfigBuilder
from graph_coder.models import GraphCoderGenerator
from graph_coder.modules import PerformerEncoder, TokenGTEncoder
from graph_coder.runners import GraphCoderGeneratorRunner


def test_config():
    params = ConfigBuilder(Path(__file__).parent / "./configs/small.yaml").load().build()

    assert isinstance(params["runner"], GraphCoderGeneratorRunner)
    assert (
        params["runner"].model.embedding
        == params["runner"].model.graph_encoder.graph_encoder.graph_feature.embedding
    )
    assert params["run"][0]["optimizer"].param_groups[0]["params"] == list(
        params["runner"].model.parameters()
    )

    for batch in params["dataset"].loaders["train"]:
        res = params["model"](batch)
        assert isinstance(res, dict)


def test_parse_config():
    root = Path(__file__).parent / "./configs/split"
    config = ConfigBuilder(root, "generator", "small", "performer").load()
    params = config.build()

    assert params["runner"] is not None
    assert params["run"] is not None
    assert isinstance(params["runner"], GraphCoderGeneratorRunner)
    assert isinstance(params["runner"].model, GraphCoderGenerator)
    assert isinstance(params["runner"].model.encoder, PerformerEncoder)
    assert isinstance(params["runner"].model.decoder, TransformerDecoder)
    assert isinstance(params["runner"].model.embedding, torch.nn.Embedding)
    assert isinstance(params["runner"].model.graph_encoder, TokenGTEncoder)
    assert isinstance(params["run"][0]["optimizer"], torch.optim.AdamW)
    assert isinstance(params["run"][0]["scheduler"], ReduceLROnPlateau)
    assert isinstance(params["run"][0]["criterion"], torch.nn.CrossEntropyLoss)
    assert isinstance(params["run"][0]["callbacks"], list)
    assert isinstance(params["run"][0]["loaders"], dict)
