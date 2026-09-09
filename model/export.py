from pathlib import Path
import json

import torch

from model.model import SimpleModel
from serving.model_identifier import ModelIdentifier
from serving.model_metadata import ModelMetadata


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def export_model(
    model_name: str,
    model_version: str,
) -> None:
    identifier = ModelIdentifier(
        name=model_name,
        version=model_version,
    )

    model_dir = ARTIFACTS_DIR / identifier.name / identifier.version

    model_path = model_dir / "model.pt"
    onnx_path = model_dir / "model.onnx"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = SimpleModel()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.tensor(
        [[0.0, 0.0]],
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["inputs"],
        output_names=["predictions"],
        dynamic_axes={
            "inputs": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
    )

    metadata = ModelMetadata(
        name=identifier.name,
        version=identifier.version,
        format="onnx",
        backend="onnx",
    )

    metadata_path.write_text(
        json.dumps(
            {
                "name": metadata.name,
                "version": metadata.version,
                "format": metadata.format,
                "backend": metadata.backend,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"ONNX model exported to: {onnx_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    export_model(
        model_name="xor",
        model_version="v1",
    )