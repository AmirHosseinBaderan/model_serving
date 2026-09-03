from pathlib import Path

from .loader import ModelLoader
from .predictor import Predictor

class ModelService:
    def __init__(self,model_path:str | Path):
        self.model_path = Path(model_path)

        self._predictor:Predictor|None = None

    def start(self):
        loader = ModelLoader(self.model_path)

        model = loader.load()
        self._predictor = Predictor(model)

    def predict(self,inputs:list[list[float]]) -> list[float]:
        if self._predictor is None:
            raise RuntimeError(
                "Modle service has not been started. "
            )

        return self._predictor.predict(inputs)