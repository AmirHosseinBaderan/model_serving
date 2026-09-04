from serving.exceptions import ModelNotReadyError
from serving.inference_engine import InferenceEngine
from serving.predictor import Predictor


class ModelService:

    def __init__(
        self,
        engine: InferenceEngine,
    ) -> None:
        self.engine = engine
        self._predictor = Predictor(engine)

    @property
    def is_ready(self) -> bool:
        return self.engine.is_ready

    def start(self) -> None:
        if self.is_ready:
            return

        self.engine.start()

    def predict(self, inputs: list[list[float]]) -> list[float]:
        if not self.is_ready:
            raise ModelNotReadyError(
                "Model service is not ready."
            )

        return self._predictor.predict(inputs)