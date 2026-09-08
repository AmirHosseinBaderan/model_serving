from tracking.mlflow_tracker import MLflowExperimentTracker
from model.train import train


def main() -> None:
    tracker = MLflowExperimentTracker(
        tracking_uri="http://127.0.0.1:5000"
    )

    try:
        tracker.get_experiment("xor")
    except ValueError:
        tracker.create_experiment("xor")

    train(
        tracker=tracker,
        experiment_name="xor",
    )


if __name__ == "__main__":
    main()