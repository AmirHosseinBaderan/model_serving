from fastapi.testclient import TestClient

from api.app import create_app
from serving.server import ModelServer
from tests.fakes import FakeInferenceEngine,FailingInferenceEngine


def create_test_client(
    engine: FakeInferenceEngine | None = None,
) -> TestClient:
    engine = engine or FakeInferenceEngine(
        predictions=[0.0, 1.0, 1.0, 2.0],
    )

    server = ModelServer(engine)

    return TestClient(
        create_app(server),
        raise_server_exceptions=False,
    )


def test_health() -> None:
    with create_test_client() as client:
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
        }


def test_ready() -> None:
    with create_test_client() as client:
        response = client.get("/api/v1/ready")

        assert response.status_code == 200
        assert response.json() == {
            "ready": True,
        }


def test_predict() -> None:
    with create_test_client() as client:
        response = client.post(
            "/api/v1/predict",
            json={
                "inputs": [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "predictions": [
                0.0,
                1.0,
                1.0,
                2.0,
            ],
        }
        
def test_predict_invalid_input() -> None:
    with create_test_client() as client:
        response = client.post(
            "/api/v1/predict",
            json={
                "inputs": []
            },
        )

        assert response.status_code == 422
        
def test_ready_when_model_not_ready() -> None:
    engine = FakeInferenceEngine(
        auto_start=False,
    )

    server = ModelServer(engine)

    with TestClient(
        create_app(server),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/v1/ready")

        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "code": "MODEL_NOT_READY",
                "message": "Model is not ready.",
            }
        }
        

def test_predict_inference_failure() -> None:
    engine = FailingInferenceEngine()

    server = ModelServer(engine)

    with TestClient(
        create_app(server),
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/api/v1/predict",
            json={
                "inputs": [
                    [0, 0],
                ]
            },
        )

        assert response.status_code == 500

        assert response.json() == {
            "error": {
                "code": "INFERENCE_FAILED",
                "message": "Inference failed.",
            }
        }