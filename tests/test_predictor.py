from pathlib import Path

from serving.predictor import Predictor
from serving.pytorch_engine import PyTorchEngine


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.pt"
)


def test_predictor():
    engine = PyTorchEngine(MODEL_PATH)

    engine.start()

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