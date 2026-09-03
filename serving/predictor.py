import torch 
from model.model import SimpleModel


class Predictor:
    def __init__(self,model:SimpleModel):
        self.model = model

    def predict(self,inputs:list[list[float]]) -> list[float]:
        tensor = torch.tensor(
            inputs,
            dtype=torch.float32
        )

        with torch.inference_mode():
            predictions = self.model(tensor)
        
        return predictions.squeeze(-1).tolist()