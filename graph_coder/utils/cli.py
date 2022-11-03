import pathlib


def expand_user(path):
    """Expand user path."""
    return pathlib.Path(path).expanduser()
