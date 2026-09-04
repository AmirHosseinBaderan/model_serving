import numpy as np

from serving.inference_engine import InferenceEngine


class FakeInferenceEngine(InferenceEngine):

    def __init__(
        self,
        predictions: list[float] | None = None,
    ) -> None:
        self._started = False
        self.predictions = predictions or [0.0]

    @property
    def is_ready(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(
                "Fake engine is not started."
            )

        return np.asarray(
            self.predictions,
            dtype=np.float32,
        ).reshape(-1, 1)