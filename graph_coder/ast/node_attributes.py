from dataclasses import dataclass, field, asdict

from graph_coder.ast.utils import empty_token


@dataclass
class NodeAttributes:
    code_name: str = field(default_factory=empty_token)
    import_name: str = field(default_factory=empty_token)
    import_as: str = field(default_factory=empty_token)
    type_name: str = field(default_factory=empty_token)
    docstring: str = field(default_factory=empty_token)
    return_type: str = field(default_factory=empty_token)

    def to_dict(self, ctx):
        return {
            k: ctx.add_string(v) if isinstance(v, str) else v
            for k, v in asdict(self).items()
        }
