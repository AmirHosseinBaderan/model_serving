import numpy as np

from serving.inference_engine import InferenceEngine


class Predictor:

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    def predict(self, inputs: list[list[float]]) -> list[float]:
        array = np.asarray(
            inputs,
            dtype=np.float32,
        )

        predictions = self.engine.predict(array)

        return predictions.squeeze(-1).tolist()