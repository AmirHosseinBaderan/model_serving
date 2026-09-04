from serving.config import AppConfig
from serving.onnx_engine import ONNXEngine
from serving.server import ModelServer


def create_server(
    config: AppConfig | None = None,
) -> ModelServer:

    config = config or AppConfig()

    if config.model_backend == "onnx":
        engine = ONNXEngine(config.model_path)

    else:
        raise ValueError(
            f"Unsupported model backend: {config.model_backend}"
        )

    return ModelServer(engine)