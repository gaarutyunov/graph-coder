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
