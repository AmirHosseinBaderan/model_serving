import pytest

from serving.model_metadata import ModelMetadata


def test_model_metadata():
    metadata = ModelMetadata(
        name="xor",
        version="v1",
        format="onnx",
        backend="onnxruntime",
    )

    assert metadata.name == "xor"
    assert metadata.version == "v1"
    assert metadata.format == "onnx"
    assert metadata.backend == "onnxruntime"


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "version",
        "format",
        "backend",
    ],
)
def test_model_metadata_rejects_empty_values(field):
    values = {
        "name": "xor",
        "version": "v1",
        "format": "onnx",
        "backend": "onnxruntime",
    }

    values[field] = ""

    with pytest.raises(ValueError):
        ModelMetadata(**values)