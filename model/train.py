from pathlib import Path

import torch
from torch import nn

from model.model import SimpleModel


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"


def train() -> None:
    torch.manual_seed(42)

    model = SimpleModel()

    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    y = torch.tensor(
        [
            [0.0],
            [1.0],
            [1.0],
            [2.0],
        ],
        dtype=torch.float32,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    model.train()

    for epoch in range(1000):
        prediction = model(x)

        loss = criterion(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:04d} | "
                f"Loss: {loss.item():.6f}"
            )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()