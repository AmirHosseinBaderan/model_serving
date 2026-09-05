from pathlib import Path

from serving.model_artifact import ModelArtifact
from serving.model_identifier import ModelIdentifier
from serving.model_metadata import ModelMetadata


def test_model_artifact():
    identifier = ModelIdentifier(
        name="xor",
        version="v1",
    )

    metadata = ModelMetadata(
        name="xor",
        version="v1",
        format="onnx",
        backend="onnxruntime",
    )

    model_path = Path(
        "model/artifacts/xor/v1/model.onnx"
    )

    artifact = ModelArtifact(
        identifier=identifier,
        model_path=model_path,
        metadata=metadata,
    )

    assert artifact.identifier == identifier
    assert artifact.model_path == model_path
    assert artifact.metadata == metadata