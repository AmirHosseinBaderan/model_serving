from dataclasses import dataclass,field

@dataclass(frozen=True)
class Experiment:
    name: str
    runs: list[str] = field(default_factory=list)