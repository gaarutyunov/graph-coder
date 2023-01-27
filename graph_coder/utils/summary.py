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
import sys
from typing import Union, Any, Iterable, Mapping, Optional

import torch
from catalyst.utils import get_device
from torch import nn
from torchinfo import ModelStatistics, summary as info_summary, Mode, Verbosity
from torchinfo.formatting import FormattingOptions
from torchinfo.layer_info import LayerInfo
from torchinfo.torchinfo import (
    INPUT_DATA_TYPE,
    DEFAULT_COLUMN_NAMES,
    DEFAULT_ROW_SETTINGS,
    process_input as info_process_input,
    CORRECTED_INPUT_DATA_TYPE,
    forward_pass as info_forward_pass,
    apply_hooks,
    set_children_layers,
    add_missing_container_layers,
)

from graph_coder.data import GraphCoderBatch


def summary(
    model: nn.Module,
    input_data: Union[INPUT_DATA_TYPE, GraphCoderBatch, None],
    col_width: int = 25,
    depth: int = 3,
    **kwargs: Any,
) -> ModelStatistics:
    if not isinstance(input_data, GraphCoderBatch):
        return info_summary(
            model, None, input_data, col_width=col_width, depth=depth, **kwargs
        )
    columns = DEFAULT_COLUMN_NAMES
    rows = DEFAULT_ROW_SETTINGS
    model_mode = Mode.EVAL
    # pylint: disable=no-member
    verbose = 0 if hasattr(sys, "ps1") and sys.ps1 else 1
    device = get_device()

    x, correct_input_size = process_input(input_data, device)
    summary_list = forward_pass(model, x, None, device, model_mode, **kwargs)
    formatting = FormattingOptions(depth, verbose, columns, col_width, rows)
    results = ModelStatistics(
        summary_list, None, 0, formatting
    )  # TODO: learn how to calculate total memory used
    if verbose > Verbosity.QUIET:
        print(results)
    return results


def process_input(
    input_data: Union[INPUT_DATA_TYPE, GraphCoderBatch], device: torch.device
) -> tuple[Union[Iterable[Any], Mapping[Any, Any], None, GraphCoderBatch], Any]:
    if isinstance(input_data, GraphCoderBatch):
        return input_data, [
            (input_data.batch_size, input_data.docstring_size),
            (
                input_data.batch_size,
                input_data.graph_size,
                input_data.node_data.size(-1),
            ),
            (input_data.batch_size, input_data.source_size),
        ]
    else:
        return info_process_input(input_data, None, None, device)


def forward_pass(
    model: nn.Module,
    x: Union[CORRECTED_INPUT_DATA_TYPE, GraphCoderBatch],
    batch_dim: Optional[int],
    device: Union[torch.device, str],
    mode: Mode,
    **kwargs: Any,
) -> list[LayerInfo]:
    """Perform a forward pass on the model using forward hooks."""
    model_name = model.__class__.__name__
    if not isinstance(x, GraphCoderBatch):
        return info_forward_pass(model, x, batch_dim, False, device, mode, **kwargs)

    # noinspection PyTypeChecker
    summary_list, _, hooks = apply_hooks(model_name, model, x, batch_dim)  # type: ignore[arg-type]
    if x is None:
        set_children_layers(summary_list)
        return summary_list

    saved_model_mode = model.training
    try:
        if mode == Mode.TRAIN:
            model.train()
        elif mode == Mode.EVAL:
            model.eval()
        else:
            raise RuntimeError(
                f"Specified model mode ({list(Mode)}) not recognized: {mode}"
            )

        with torch.no_grad():
            _ = model(x)
    except Exception as e:
        executed_layers = [layer for layer in summary_list if layer.executed]
        raise RuntimeError(
            "Failed to run torchinfo. See above stack traces for more details. "
            f"Executed layers up to: {executed_layers}"
        ) from e
    finally:
        if hooks:
            for pre_hook, hook in hooks.values():
                pre_hook.remove()
                hook.remove()
        model.train(saved_model_mode)

    add_missing_container_layers(summary_list)
    set_children_layers(summary_list)

    return summary_list
