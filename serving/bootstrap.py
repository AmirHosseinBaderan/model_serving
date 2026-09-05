from serving.artifact_resolver import ModelArtifactResolver
from serving.config import AppConfig
from serving.model_identifier import ModelIdentifier
from serving.onnx_engine import ONNXEngine
from serving.server import ModelServer


def create_server(
    config: AppConfig | None = None,
) -> ModelServer:

    config = config or AppConfig()

    identifier = ModelIdentifier(
        name=config.model_name,
        version=config.model_version,
    )

    resolver = ModelArtifactResolver(
        artifacts_root=config.model_artifacts_root,
    )

    model_path = resolver.resolve(identifier)

    if config.model_backend == "onnx":
        engine = ONNXEngine(model_path)

    else:
        raise ValueError(
            f"Unsupported model backend: {config.model_backend}"
        )

    return ModelServer(engine)