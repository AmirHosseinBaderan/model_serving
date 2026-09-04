from pathlib import Path

import pytest

from serving.onnx_engine import ONNXEngine
from serving.server import ModelServer


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.onnx"
)


def create_server() -> ModelServer:
    engine = ONNXEngine(MODEL_PATH)

    return ModelServer(engine)


def test_server_is_not_running_initially():
    server = create_server()

    assert server.is_running is False
    assert server.is_ready is False


def test_server_starts_and_loads_model():
    server = create_server()

    server.start()

    assert server.is_running is True
    assert server.is_ready is True


def test_server_predicts():
    server = create_server()

    server.start()

    result = server.predict(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert len(result) == 4


def test_server_cannot_predict_before_start():
    server = create_server()

    with pytest.raises(RuntimeError):
        server.predict([[1.0, 1.0]])


def test_server_stop():
    server = create_server()

    server.start()

    assert server.is_running is True

    server.stop()

    assert server.is_running is False


def test_server_start_is_idempotent():
    server = create_server()

    server.start()

    service = server.service

    server.start()

    assert server.service is service
    assert server.is_running is True
    assert server.is_ready is True