from pathlib import Path

import json
import pytest

from serving.artifact_resolver import ModelArtifactResolver
from serving.model_artifact import ModelArtifact
from serving.model_identifier import ModelIdentifier


def test_resolve_model_artifact(tmp_path: Path) -> None:
    model_path = (
        tmp_path
        / "xor"
        / "v1"
        / "model.onnx"
    )

    metadata_path = (
        tmp_path
        / "xor"
        / "v1"
        / "metadata.json"
    )

    model_path.parent.mkdir(
        parents=True,
    )

    model_path.touch()

    metadata_path.write_text(
        json.dumps(
            {
                "name": "xor",
                "version": "v1",
                "format": "onnx",
                "backend": "onnxruntime",
            }
        ),
        encoding="utf-8",
    )

    resolver = ModelArtifactResolver(tmp_path)

    identifier = ModelIdentifier(
        name="xor",
        version="v1",
    )

    result = resolver.resolve(identifier)

    assert isinstance(result, ModelArtifact)
    assert result.identifier == identifier
    assert result.model_path == model_path
    assert result.metadata.name == "xor"
    assert result.metadata.version == "v1"
    assert result.metadata.format == "onnx"
    assert result.metadata.backend == "onnxruntime"


def test_resolve_missing_model_artifact(
    tmp_path: Path,
) -> None:
    resolver = ModelArtifactResolver(tmp_path)

    identifier = ModelIdentifier(
        name="xor",
        version="v1",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found: xor:v1",
    ):
        resolver.resolve(identifier)
        
def test_resolve_rejects_metadata_identifier_mismatch(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "xor" / "v1" / "model.onnx"
    metadata_path = tmp_path / "xor" / "v1" / "metadata.json"

    model_path.parent.mkdir(parents=True)
    model_path.touch()

    metadata_path.write_text(
        json.dumps({
            "name": "xor",
            "version": "v2",
            "format": "onnx",
            "backend": "onnxruntime",
        }),
        encoding="utf-8",
    )

    resolver = ModelArtifactResolver(tmp_path)

    identifier = ModelIdentifier(
        name="xor",
        version="v1",
    )

    with pytest.raises(
        ValueError,
        match="Model metadata does not match requested identifier",
    ):
        resolver.resolve(identifier)