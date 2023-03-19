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
#
#  Copyright (c) OpenAI (https://openai.com)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd
from catalyst.core import Engine
from human_eval.evaluation import evaluate_functional_correctness
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from graph_coder.data import GraphCoderBatch
from graph_coder.datasets import HumanEvalDataset
from graph_coder.utils import print_rank0

from .generator import GraphCoderGeneratorRunner, TMM


class HumanEvalRunner(GraphCoderGeneratorRunner):
    def __init__(
        self,
        model: TMM,
        log_path: Union[str, PathLike],
        problem_file: Union[str, PathLike],
        tokenizer: PreTrainedTokenizerBase,
        eos_token_id: int,
        vocab_size: int,
        model_path: str = "model.best.pth",
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(model, eos_token_id, vocab_size, *args, **kwargs)
        self.log_path = Path(log_path)
        self.resume = self.log_path / model_path
        self.tokenizer = tokenizer
        self.problem_file = Path(problem_file).expanduser()
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        return self.model.generate(
            self.num_samples,
            self.temperature,
            self.top_k,
            self.top_p,
            self.repetition_penalty,
            **batch,
        )

    def evaluate(
        self, loader: Optional[DataLoader] = None, engine: Optional[Engine] = None
    ):
        loader = loader or self.loaders["infer"]
        assert loader is not None

        eval_dir = self.log_path / "human_eval"
        eval_dir.mkdir(exist_ok=True)

        df = pd.DataFrame(columns=["task_id", "completion"])

        dataset: Union[Subset, Dataset] = loader.dataset

        if isinstance(dataset, Subset):
            he_dataset: HumanEvalDataset = dataset.dataset  # type: ignore[assignment]
        else:
            he_dataset = dataset  # type: ignore[assignment]

        for i, result in enumerate(
            tqdm(
                self.predict_loader(
                    loader=loader, resume=str(self.resume), engine=engine
                ),
                desc="Generating completions...",
                total=len(loader),
            )
        ):
            if isinstance(dataset, Subset):
                example = dataset.dataset[dataset.indices[i]]
            else:
                example = dataset[i]
            batch = GraphCoderBatch.from_dict(he_dataset.collate_fn([example]))  # type: ignore[misc]

            for j in range(self.num_samples):
                source = result["source"][j, batch.source.size(1) :]
                decoded_source = self.tokenizer.decode(source)
                new_df = pd.DataFrame(
                    columns=df.columns, data=[[example.task_id, decoded_source]]
                )
                df = pd.concat([df, new_df])

        sample_file = eval_dir / "samples.jsonl"

        df.to_json(sample_file, orient="records", lines=True)
        results = evaluate_functional_correctness(
            str(sample_file),
            problem_file=str(self.problem_file),
            ignore_incomplete=True,
        )
        print_rank0(json.dumps(results, indent=4))
