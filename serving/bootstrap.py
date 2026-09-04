from pathlib import Path

from .onnx_engine import ONNXEngine
from .server import ModelServer

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "artifacts"
    / "model.onnx"
)


def create_server() -> ModelServer:
    engine = ONNXEngine(MODEL_PATH)

    return ModelServer(engine)