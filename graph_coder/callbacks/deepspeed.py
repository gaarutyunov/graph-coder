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
from typing import Any, Optional

from catalyst import dl
from catalyst.core import IRunner
from deepspeed import DeepSpeedEngine


class DeepSpeedCheckpointCallback(dl.CheckpointCallback):
    def __init__(
        self,
        logdir: str,
        loader_key: Optional[str] = None,
        metric_key: Optional[str] = None,
        minimize: Optional[bool] = None,
        topk: int = 1,
        save_last: bool = True,
        resume_model: Optional[str] = None,
        resume_runner: Optional[str] = None,
        load_best_on_end: bool = False,
    ):
        super().__init__(
            logdir,
            loader_key,
            metric_key,
            minimize,
            topk,
            "model",
            save_last,
            False,
            resume_model,
            resume_runner,
            load_best_on_end,
        )

    def _save(self, runner: IRunner, obj: DeepSpeedEngine, logprefix: str) -> str:
        assert isinstance(
            obj, DeepSpeedEngine
        ), "obj must be a :py:mod:`deepspeed.DeepSpeedEngine`"

        obj.save_checkpoint(self.logdir, save_latest=self.save_last)
        tag = f"global_step{obj.global_steps}"

        return obj._get_ckpt_name(self.logdir, tag)

    def _load(
        self,
        runner: IRunner,
        resume_logpath: Optional[Any] = None,
        resume_model: Optional[str] = None,
        resume_runner: Optional[str] = None,
    ):
        assert isinstance(
            runner.model, DeepSpeedEngine
        ), "runner.model must be a :py:mod:`deepspeed.DeepSpeedEngine`"

        runner.model.load_checkpoint(self.logdir, tag=resume_logpath)
