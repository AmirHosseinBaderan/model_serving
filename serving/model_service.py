from pathlib import Path

from .loader import ModelLoader
from .predictor import Predictor

class ModelService:
    def __init__(self,model_path:str | Path):
        self.model_path = Path(model_path)

        self._predictor:Predictor|None = None

    @property
    def is_ready(self)-> bool:
        return self._predictor is not None

    def start(self):
        if self.is_ready:
            return
        
        loader = ModelLoader(self.model_path)

        model = loader.load()
        self._predictor = Predictor(model)

    def predict(self,inputs:list[list[float]]) -> list[float]:
        if not self.is_ready:
            raise RuntimeError(
                "Modle service has not been started. "
            )

        return self._predictor.predict(inputs)