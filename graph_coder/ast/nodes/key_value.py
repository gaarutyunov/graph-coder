import ast
import typing

from graph_coder.ast.utils import wrap


class KeyValue(ast.expr):
    def __init__(self, key, value, *args: typing.Any, **kwargs: typing.Any):
        super().__init__(*args, **kwargs)
        self.key = wrap(key)
        self.value = wrap(value)

    _fields = ("key", "value")

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        if not hasattr(self, "key"):
            raise NotImplementedError
        if isinstance(self.value, ast.Name) and isinstance(self.key, ast.Constant):
            return f"{self.value.id}.{self.key.value}"
        if isinstance(self.value, ast.Constant) and isinstance(self.key, ast.Constant):
            return f"{self.value.value}.{self.key.value}"
        if isinstance(self.value, KeyValue) and isinstance(self.key, ast.Constant):
            return f"{self.value!s}.{self.key.value}"
        else:
            return f"{self.value!r}[{self.key!r}]"
