from pathlib import Path

from serving.model_service import ModelService
from serving.pytorch_engine import PyTorchEngine


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.pt"
)


class ModelServer:

    def __init__(self, model_path: str | Path) -> None:
        engine = PyTorchEngine(model_path)

        self.service = ModelService(engine)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_ready(self) -> bool:
        return self.service.is_ready

    def start(self) -> None:
        if self._running:
            return

        self.service.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

    def predict(
        self,
        inputs: list[list[float]],
    ) -> list[float]:
        if not self.is_running:
            raise RuntimeError(
                "Model server is not running."
            )

        return self.service.predict(inputs)


def create_server() -> ModelServer:
    return ModelServer(MODEL_PATH)