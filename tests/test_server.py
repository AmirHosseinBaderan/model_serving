from pathlib import Path

import pytest

from serving.server import ModelServer


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "artifacts"
    / "model.pt"
)


def test_server_is_not_running_initially():
    server = ModelServer(MODEL_PATH)

    assert server.is_running is False
    assert server.is_ready is False


def test_server_starts_and_loads_model():
    server = ModelServer(MODEL_PATH)

    server.start()

    assert server.is_running is True
    assert server.is_ready is True


def test_server_predicts():
    server = ModelServer(MODEL_PATH)

    server.start()

    result = server.predict(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert len(result) == 2


def test_server_cannot_predict_before_start():
    server = ModelServer(MODEL_PATH)

    with pytest.raises(RuntimeError):
        server.predict([[1.0, 1.0]])


def test_server_stop():
    server = ModelServer(MODEL_PATH)

    server.start()
    server.stop()

    assert server.is_running is False


def test_server_start_is_idempotent():
    server = ModelServer(MODEL_PATH)

    server.start()

    service = server.service
    predictor = service._predictor

    server.start()

    assert server.service is service
    assert server.service._predictor is predictor