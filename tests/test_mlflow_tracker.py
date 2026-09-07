import mlflow

from tracking.mlflow_tracker import MLflowExperimentTracker
from tracking.run import RunStatus


def test_create_experiment() -> None:
    tracker = MLflowExperimentTracker()

    experiment = tracker.create_experiment("xor")

    assert experiment.name == "xor"


def test_start_run() -> None:
    tracker = MLflowExperimentTracker()

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    assert run.experiment_name == "xor"
    assert run.id


def test_log_parameter() -> None:
    tracker = MLflowExperimentTracker()

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_parameter(run, "learning_rate", 0.01)

    assert run.parameters["learning_rate"] == 0.01


def test_log_metric() -> None:
    tracker = MLflowExperimentTracker()

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_metric(run, "loss", 0.5, step=1)

    assert run.latest_metrics["loss"] == 0.5


def test_finish_run() -> None:
    tracker = MLflowExperimentTracker()

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.finish_run(run)

    assert run.status == RunStatus.COMPLETED
    assert run.finished_at is not None