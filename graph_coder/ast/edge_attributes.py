from dataclasses import dataclass, asdict


@dataclass
class EdgeAttributes:
    def to_dict(self, ctx):
        return {
            k: ctx.add_string(v) if isinstance(v, str) else v
            for k, v in asdict(self).items()
        }
