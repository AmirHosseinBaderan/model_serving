from pathlib import Path

import pytest

from serving.artifact_resolver import ModelArtifactResolver
from serving.model_identifier import ModelIdentifier


def test_resolve_model_artifact(tmp_path: Path) -> None:
    model_path = (
        tmp_path
        / "xor"
        / "v1"
        / "model.onnx"
    )

    model_path.parent.mkdir(
        parents=True,
    )

    model_path.touch()

    resolver = ModelArtifactResolver(tmp_path)

    identifier = ModelIdentifier(
        name="xor",
        version="v1",
    )

    result = resolver.resolve(identifier)

    assert result == model_path


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