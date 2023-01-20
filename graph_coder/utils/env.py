#  Copyright 2023 German Arutyunov
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
import sys


def check_ipython() -> bool:
    try:
        get_ipython = sys.modules["IPython"].get_ipython

        if get_ipython is None:
            return False
        else:
            _ = str(get_ipython())
            return True
    except:
        return False


def run_async(func):
    import asyncio

    if check_ipython():
        import nest_asyncio

        nest_asyncio.apply()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(func)
