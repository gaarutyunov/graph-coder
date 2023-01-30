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
import os
import pathlib
import sys
from datetime import datetime
from typing import Union

import aiofiles

from graph_coder.logger.logger import ILogger


class AsyncLogger(ILogger):
    def __init__(self, f: Union[str, os.PathLike]):
        self.f = pathlib.Path(f).expanduser()

    async def log(self, level: str, msg: str):
        level = f"[{level}]"
        async with aiofiles.open(self.f, "a") as log:
            await log.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {level:8s} {msg}\n"
            )

    async def info(self, msg: str):
        await self.log("INFO", msg)

    async def warn(self, msg: str):
        await self.log("WARN", msg)

    async def error(self, msg: str):
        await self.log("ERROR", msg)

    async def debug(self, msg: str):
        await self.log("DEBUG", msg)

    async def fatal(self, msg: str):
        await self.log("FATAL", msg)
        sys.exit(1)
