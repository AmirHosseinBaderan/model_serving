from pathlib import Path

import numpy as np

from serving.onnx_engine import ONNXEngine


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.onnx"
)


def test_onnx_engine_predict():

    engine = ONNXEngine(MODEL_PATH)

    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = engine.predict(inputs)

    assert result.shape == (4, 1)