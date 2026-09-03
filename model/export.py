from pathlib import Path
import torch
from .model import SimpleModel

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "model.pt"
ONNX_PATH = BASE_DIR / "artifacts" / "model.onnx"

def export_model():
    model = SimpleModel()
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
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
        ONNX_PATH,
        input_names=['inputs'],
        output_names=['predictions'],
        dynamic_axes={
            "inputs":{0:"batch_size"},
            "predictions":{0:"batch_size"}
        }
    )
    
    print(f"ONNX model exported to: {ONNX_PATH}")
    
if __name__ == "__main__":
    export_model()