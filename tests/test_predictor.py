from pathlib import Path

from serving.loader import ModelLoader
from serving.predictor import Predictor


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "artifacts"
    / "model.pt"
)


def test_predictor():
    loader = ModelLoader(MODEL_PATH)

    model = loader.load()

    predictor = Predictor(model)

    result = predictor.predict(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert len(result) == 4

    print(result)