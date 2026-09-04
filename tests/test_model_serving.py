from pathlib import Path

import pytest

from serving.model_service import ModelService
from serving.pytorch_engine import PyTorchEngine


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.pt"
)


def create_service() -> ModelService:
    engine = PyTorchEngine(MODEL_PATH)

    return ModelService(engine)


def test_service_is_not_ready_before_start():
    service = create_service()

    assert service.is_ready is False


def test_service_is_ready_after_start():
    service = create_service()

    service.start()

    assert service.is_ready is True


def test_service_requires_start_before_prediction():
    service = create_service()

    with pytest.raises(RuntimeError):
        service.predict([[1.0, 1.0]])


def test_service_predicts_after_start():
    service = create_service()

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
    service = create_service()

    service.start()

    engine = service.engine

    service.start()

    assert service.engine is engine
    assert service.is_ready is True