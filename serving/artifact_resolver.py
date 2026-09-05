from pathlib import Path

from serving.model_artifact import ModelArtifact
from serving.model_identifier import ModelIdentifier
from serving.model_metadata_loader import ModelMetadataLoader


class ModelArtifactResolver:

    def __init__(
        self,
        artifacts_root: str | Path,
    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.metadata_loader = ModelMetadataLoader()

    def resolve(
        self,
        identifier: ModelIdentifier,
    ) -> ModelArtifact:

        artifact_directory = (
            self.artifacts_root
            / identifier.name
            / identifier.version
        )

        model_path = (
            artifact_directory
            / "model.onnx"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {identifier}"
            )

        metadata_path = (
            artifact_directory
            / "metadata.json"
        )

        metadata = self.metadata_loader.load(
            metadata_path,
        )
        
        if (
            metadata.name != identifier.name
            or metadata.version != identifier.version
        ):
            raise ValueError(
                "Model metadata does not match requested identifier"
            )

        return ModelArtifact(
            identifier=identifier,
            model_path=model_path,
            metadata=metadata,
        )