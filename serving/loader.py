from pathlib import Path
import torch
from model.model import SimpleModel

class ModelLoader:
    def __init__(self,model_path:str | Path):
        self.model_path = Path(model_path)

    def load(self)-> SimpleModel:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found : {self.model_path}"
            )

        model = SimpleModel()
        state_dict = torch.load(
            self.model_path,
            map_location="cpu"
        )

        model.load_state_dict(state_dict)
        model.eval()

        return model