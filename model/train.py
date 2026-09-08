from pathlib import Path

import torch
from torch import nn

from model.model import SimpleModel
from tracking.tracker import ExperimentTracker


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"


def train(
    tracker: ExperimentTracker | None = None,
    experiment_name: str = "xor",
    model_path: Path = MODEL_PATH,
    epochs: int = 1000,
) -> None:
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

    run = None

    if tracker is not None:
        try:
            tracker.get_experiment(experiment_name)
        except ValueError:
            tracker.create_experiment(experiment_name)

        run = tracker.start_run(experiment_name)

        tracker.log_parameter(
            run,
            "learning_rate",
            0.01,
        )

        tracker.log_parameter(
            run,
            "epochs",
            epochs,
        )

        tracker.log_parameter(
            run,
            "optimizer",
            "Adam",
        )

        tracker.log_metadata(
            run,
            "seed",
            42,
        )

    try:
        for epoch in range(epochs):
            prediction = model(x)

            loss = criterion(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if run is not None:
                tracker.log_metric(
                    run,
                    "train_loss",
                    loss.item(),
                    step=epoch + 1,
                )

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1:04d} | "
                    f"Loss: {loss.item():.6f}"
                )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            model_path,
        )

        if run is not None:
            tracker.log_artifact(
                run,
                str(model_path),
            )

            tracker.finish_run(run)

        print(f"\nModel saved to: {model_path}")

    except Exception:
        if run is not None:
            tracker.fail_run(run)

        raise


if __name__ == "__main__":
    train()