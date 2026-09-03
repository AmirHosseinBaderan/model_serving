import numpy as np
import torch

from model.model import SimpleModel
from .inference_engine import InferenceEngine

class PyTorchEngine(InferenceEngine):
    def __init__(self,model:SimpleModel):
        self.model = model
        
    def predict(self,inputs:np.ndarray)-> np.ndarray:
        tensor = torch.from_numpy(inputs).float()
        
        with torch.inference_mode():
            predictions = self.model(tensor)
            
        return predictions.numpy()