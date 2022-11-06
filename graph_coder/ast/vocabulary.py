import typing
from dataclasses import dataclass, field

from graph_coder.ast.utils import empty_token


@dataclass
class Vocabulary:
    counter: int = 1
    key_value: typing.Dict[int, str] = field(default_factory=lambda: {0: empty_token})
    value_key: typing.Dict[str, int] = field(default_factory=lambda: {empty_token: 0})
    value_types: typing.Dict[int, str] = field(default_factory=lambda: {0: "special"})

    def __post_init__(self):
        self.counter = len(self.key_value)

    def add(self, value: str, value_type: str = "word") -> int:
        if value in self.value_key:
            return self.value_key[value]
        self.key_value[self.counter] = value
        self.value_key[value] = self.counter
        self.value_types[self.counter] = value_type
        self.counter += 1
        return self.counter - 1

    def __len__(self):
        return len(self.key_value)
