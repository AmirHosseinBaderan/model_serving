from dataclasses import dataclass
from pathlib import Path

from serving.model_identifier import ModelIdentifier
from serving.model_metadata import ModelMetadata


@dataclass(frozen=True)
class ModelArtifact:
    identifier: ModelIdentifier
    model_path: Path
    metadata: ModelMetadata