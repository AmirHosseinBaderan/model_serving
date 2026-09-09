import argparse

from model.export import export_model
from model.train import train
from tracking.mlflow_tracker import MLflowExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export a model."
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--model-version",
        required=True,
    )

    parser.add_argument(
        "--experiment-name",
        default="xor",
    )

    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )

    args = parser.parse_args()

    tracker = MLflowExperimentTracker(
        tracking_uri=args.tracking_uri,
    )

    try:
        tracker.get_experiment(args.experiment_name)
    except ValueError:
        tracker.create_experiment(args.experiment_name)

    print("=== Training ===")

    train(
        tracker=tracker,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    print("\n=== Export ===")

    export_model(
        model_name=args.model_name,
        model_version=args.model_version,
    )

    print("\n=== Pipeline completed ===")


if __name__ == "__main__":
    main()
