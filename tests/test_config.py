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
import pathlib
import shutil
from collections import OrderedDict
from pathlib import Path

import graph_coder.config.functional as F

import torch
from catalyst.registry import REGISTRY

from graph_coder.config import ConfigBuilder
from graph_coder.models import GraphCoderGenerator
from graph_coder.modules import PerformerEncoder, TokenGTEncoder
from graph_coder.runners import GraphCoderGeneratorRunner
from torch._C._profiler import ProfilerActivity
from torch.nn import TransformerDecoder
from torch.optim.lr_scheduler import ReduceLROnPlateau


def test_config():
    params = (
        ConfigBuilder(Path(__file__).parent / "./configs/small.yaml").load().build()
    )

    assert isinstance(params["runner"], GraphCoderGeneratorRunner)
    assert (
        params["runner"].model.embedding
        == params[
            "runner"
        ].model.graph_encoder.graph_encoder.graph_feature.embedding.embedding
    )
    assert params["run"][0]["optimizer"].param_groups[0]["params"] == list(
        params["runner"].model.parameters()
    )

    for batch in params["dataset"].loaders["train"]:
        res = params["model"](**batch)
        assert isinstance(res, dict)


def test_parse_config():
    root = Path(__file__).parent / "./configs/split"
    config = ConfigBuilder(root, "generator", "tiny", "performer").load()
    params = config.build()

    assert params["runner"] is not None
    assert params["run"] is not None
    assert isinstance(params["runner"], GraphCoderGeneratorRunner)
    assert isinstance(params["runner"].model, GraphCoderGenerator)
    assert isinstance(params["runner"].model.text_encoder, PerformerEncoder)
    assert isinstance(params["runner"].model.code_encoder, PerformerEncoder)
    assert isinstance(params["runner"].model.decoder, TransformerDecoder)
    assert isinstance(params["runner"].model.embedding, torch.nn.Embedding)
    assert isinstance(params["runner"].model.graph_encoder, TokenGTEncoder)
    assert isinstance(params["run"][0]["optimizer"], torch.optim.AdamW)
    assert isinstance(params["run"][0]["scheduler"], ReduceLROnPlateau)
    assert isinstance(params["run"][0]["criterion"], torch.nn.CrossEntropyLoss)
    assert isinstance(params["run"][0]["callbacks"], list)
    assert isinstance(params["run"][0]["loaders"], dict)


def test_get_activity():
    config = OrderedDict(
        activities=[
            OrderedDict(
                _target_="graph_coder.config.F.get_activity", idx=0, _mode_="call"
            ),
            OrderedDict(
                _target_="graph_coder.config.F.get_activity", idx=1, _mode_="call"
            ),
        ]
    )

    activities = REGISTRY.get_from_params(**config)

    assert activities["activities"][0] == ProfilerActivity.CPU
    assert activities["activities"][1] == ProfilerActivity.CUDA


def test_log_path():
    root = pathlib.Path(__file__).parent / "logs" / "test_logs"

    shutil.rmtree(root, ignore_errors=True)

    for i in range(0, 20):
        _ = F.get_log_path(str(root))

    for i in range(1, 21):
        assert (root / f"version{i}").exists()

    assert Path(F.get_log_path(str(root), create=False)).stem == "version20"

    shutil.rmtree(root)
