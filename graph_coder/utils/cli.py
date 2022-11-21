import pathlib
import typing


def expand_user(path):
    """Expand user path."""
    return pathlib.Path(path).expanduser()


def split_string(sep=","):
    """Split string."""
    def inner(s):
        if s == "":
            return []
        return s.split(sep)

    return inner


def to_int(s: typing.List[str]) -> typing.List[int]:
    """Convert to int."""
    return [int(i) for i in s]
