import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    model_backend: str
    model_path: str

    @classmethod
    def from_environment(cls) -> "AppConfig":
        backend = os.getenv(
            "MODEL_BACKEND",
            "onnx",
        ).lower()

        model_path = os.getenv(
            "MODEL_PATH",
            "/app/model/artifacts/model.onnx",
        )

        return cls(
            model_backend=backend,
            model_path=model_path,
        )