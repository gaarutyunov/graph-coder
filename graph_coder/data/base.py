import dataclasses
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class BaseExample(Generic[T]):
    """Base class for examples."""

    source: str
    graph: T
    docstring: str
