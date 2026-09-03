from pathlib import Path

from model.model import SimpleModel
from serving.loader import ModelLoader
from serving.predictor import Predictor
from serving.pytorch_engine import PyTorchEngine


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.pt"
)


def test_predictor():
    loader = ModelLoader(MODEL_PATH)

    model = loader.load()

    engine = PyTorchEngine(model)

    predictor = Predictor(engine)

    result = predictor.predict(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert len(result) == 4