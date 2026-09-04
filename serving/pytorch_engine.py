from pathlib import Path

import numpy as np
import torch

from model.model import SimpleModel
from serving.inference_engine import InferenceEngine


class PyTorchEngine(InferenceEngine):

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model: SimpleModel | None = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def start(self) -> None:
        if self.is_ready:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        model = SimpleModel()

        state_dict = torch.load(
            self.model_path,
            map_location="cpu",
        )

        model.load_state_dict(state_dict)
        model.eval()

        self.model = model

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(
                "PyTorch engine is not started."
            )

        tensor = torch.from_numpy(inputs).float()

        with torch.inference_mode():
            predictions = self.model(tensor)

        return predictions.numpy()