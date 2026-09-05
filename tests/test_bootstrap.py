from pathlib import Path

from serving.bootstrap import create_server
from serving.config import AppConfig

import pytest


def test_create_server_resolves_versioned_model(
    tmp_path: Path,
) -> None:
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

    config = AppConfig(
        model_backend="onnx",
        model_name="xor",
        model_version="v1",
        model_artifacts_root=str(tmp_path),
    )

    server = create_server(config)

    assert server is not None
    assert server.service.engine.model_path == model_path
    
def test_create_server_rejects_unsupported_backend() -> None:
    config = AppConfig(
        model_backend="tensorflow",
        model_name="xor",
        model_version="v1",
        model_artifacts_root="./model/artifacts",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported model backend: tensorflow",
    ):
        create_server(config)