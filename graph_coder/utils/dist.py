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


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def print_rank0(fmt: str, *args, sep=" ", end="\n", file=None):
    if get_local_rank() in [0, -1]:
        print(fmt, *args, sep=sep, end=end, file=file)
