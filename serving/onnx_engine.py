from pathlib import Path

import numpy as np
import onnxruntime as ort

from serving.inference_engine import InferenceEngine


class ONNXEngine(InferenceEngine):

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.session: ort.InferenceSession | None = None

        self.input_name: str | None = None
        self.output_name: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    def start(self) -> None:
        if self.is_ready:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(
                "ONNX engine is not started."
            )

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: inputs,
            },
        )

        return outputs[0]