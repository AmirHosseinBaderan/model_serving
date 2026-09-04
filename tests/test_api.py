from fastapi.testclient import TestClient

from api.app import create_app
from serving.server import ModelServer
from tests.fakes import FakeInferenceEngine


def create_test_client() -> TestClient:
    engine = FakeInferenceEngine(
        predictions=[0.0, 1.0, 1.0, 2.0],
    )

    server = ModelServer(engine)

    return TestClient(
        create_app(server)
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