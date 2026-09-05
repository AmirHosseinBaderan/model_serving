import json

import pytest

from serving.model_metadata_loader import ModelMetadataLoader


def test_load_model_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.json"

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

    loader = ModelMetadataLoader()

    metadata = loader.load(metadata_path)

    assert metadata.name == "xor"
    assert metadata.version == "v1"
    assert metadata.format == "onnx"
    assert metadata.backend == "onnxruntime"


def test_load_missing_metadata(tmp_path):
    loader = ModelMetadataLoader()

    metadata_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Metadata file not found",
    ):
        loader.load(metadata_path)