import numpy as np

from serving.inference_engine import InferenceEngine


class FakeInferenceEngine(InferenceEngine):

    def __init__(
        self,
        predictions: list[float] | None = None,
        auto_start: bool = True,
    ) -> None:
        self._started = False
        self.predictions = predictions or [0.0]
        self.auto_start = auto_start

    @property
    def is_ready(self) -> bool:
        return self._started

    def start(self) -> None:
        if self.auto_start:
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
        
class FailingInferenceEngine(FakeInferenceEngine):

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        raise RuntimeError("Internal inference failure.")