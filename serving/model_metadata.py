from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    format: str
    backend: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name cannot be empty.")

        if not self.version:
            raise ValueError("Model version cannot be empty.")

        if not self.format:
            raise ValueError("Model format cannot be empty.")

        if not self.backend:
            raise ValueError("Model backend cannot be empty.")