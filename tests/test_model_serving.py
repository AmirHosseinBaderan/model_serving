from pathlib import Path

import pytest

from serving.model_service import ModelService


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "artifacts"
    / "model.pt"
)


def test_service_is_not_ready_before_start():
    service = ModelService(MODEL_PATH)

    assert service.is_ready is False


def test_service_is_ready_after_start():
    service = ModelService(MODEL_PATH)

    service.start()

    assert service.is_ready is True


def test_service_requires_start_before_prediction():
    service = ModelService(MODEL_PATH)

    with pytest.raises(RuntimeError):
        service.predict([[1.0, 1.0]])


def test_service_predicts_after_start():
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


def test_start_is_idempotent():
    service = ModelService(MODEL_PATH)

    service.start()
    first_predictor = service._predictor

    service.start()
    second_predictor = service._predictor

    assert first_predictor is second_predictor