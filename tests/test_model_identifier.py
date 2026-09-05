import pytest

from serving.model_identifier import ModelIdentifier


def test_model_identifier() -> None:
    identifier = ModelIdentifier(
        name="xor",
        version="v1",
    )

    assert identifier.name == "xor"
    assert identifier.version == "v1"
    assert str(identifier) == "xor:v1"


def test_model_identifier_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ModelIdentifier(
            name="",
            version="v1",
        )


def test_model_identifier_rejects_empty_version() -> None:
    with pytest.raises(ValueError):
        ModelIdentifier(
            name="xor",
            version="",
        )