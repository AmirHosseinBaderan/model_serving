from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model.model import SimpleModel


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "xor"
    / "v1"
    / "model.onnx"
)


def test_onnx_model_can_load():
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    assert session is not None


def test_onnx_model_can_predict():
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    outputs = session.run(
        ["predictions"],
        {"inputs": inputs},
    )

    predictions = outputs[0]

    assert predictions.shape == (4, 1)


def test_onnx_matches_pytorch():
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    model = SimpleModel()

    state_dict = torch.load(
        "model/artifacts/model.pt",
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    with torch.inference_mode():
        pytorch_predictions = (
            model(torch.from_numpy(inputs))
            .numpy()
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_predictions = session.run(
        ["predictions"],
        {"inputs": inputs},
    )[0]

    np.testing.assert_allclose(
        pytorch_predictions,
        onnx_predictions,
        rtol=1e-5,
        atol=1e-6,
    )