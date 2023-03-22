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
import torch
import wrapt


class _DataLoaderIterWrapper(wrapt.ObjectProxy):
    def __next__(self):
        # second item in tuple is the outputs, that we don't use in the model
        return tuple([self.__wrapped__.__next__(), tuple(torch.empty(1, 1))])


class PipeLoaderWrapper(wrapt.ObjectProxy):
    def __iter__(self):
        return _DataLoaderIterWrapper(self.__wrapped__.__iter__())
