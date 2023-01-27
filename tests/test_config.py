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

from catalyst.contrib.scripts.run import process_configs
from catalyst.registry import REGISTRY

from graph_coder.runners import GraphCoderGeneratorRunner


def test_config():
    configs = [f'{Path(__file__).parent / "./configs/small.yaml"}']
    configs = process_configs(configs)
    params = REGISTRY.get_from_params(**configs)

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
