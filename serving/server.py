from pathlib import Path

from serving.inference_engine import InferenceEngine
from serving.model_service import ModelService


class ModelServer:

    def __init__(self, engine: InferenceEngine) -> None:
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