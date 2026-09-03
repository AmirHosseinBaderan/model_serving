from pathlib import Path

import pytest

from serving.model_service import ModelService


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "artifacts"
    / "model.pt"
)


def test_service_requires_start():
    service = ModelService(MODEL_PATH)

    with pytest.raises(RuntimeError):
        service.predict([[1.0, 1.0]])


def test_service_loads_model_once_and_predicts():
    service = ModelService(MODEL_PATH)

    service.start()

    result = service.predict(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert len(result) == 4