import mlflow

from tracking.mlflow_tracker import MLflowExperimentTracker
from tracking.run import RunStatus
import pytest

def create_tracker(tmp_path):
    db_path = tmp_path / "mlflow.db"

    return MLflowExperimentTracker(
        tracking_uri=f"sqlite:///{db_path}"
    )
    
@pytest.fixture(autouse=True)
def cleanup_mlflow_run():
    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()

def test_create_experiment(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    experiment = tracker.create_experiment("xor")

    assert experiment.name == "xor"


def test_start_run(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    assert run.experiment_name == "xor"
    assert run.id


def test_log_parameter(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_parameter(run, "learning_rate", 0.01)

    assert run.parameters["learning_rate"] == 0.01


def test_log_metric(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_metric(run, "loss", 0.5, step=1)

    assert run.latest_metrics["loss"] == 0.5


def test_finish_run(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.finish_run(run)

    assert run.status == RunStatus.COMPLETED
    assert run.finished_at is not None
    
def test_parameter_is_persisted_in_mlflow(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_parameter(run, "learning_rate", 0.01)

    tracker.finish_run(run)

    mlflow_run = mlflow.get_run(run.id)

    assert mlflow_run.data.params["learning_rate"] == "0.01"
    
def test_metric_is_persisted_in_mlflow(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_metric(run, "loss", 0.25, step=1)
    tracker.log_metric(run, "loss", 0.10, step=2)

    tracker.finish_run(run)

    mlflow_run = mlflow.get_run(run.id)

    assert mlflow_run.data.metrics["loss"] == 0.10
    
def test_metadata_is_persisted_in_mlflow(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_metadata(
        run,
        "git_commit",
        "a83f21c",
    )

    tracker.finish_run(run)

    mlflow_run = mlflow.get_run(run.id)

    assert mlflow_run.data.tags["git_commit"] == "a83f21c"
    
def test_artifact_is_persisted_in_mlflow(tmp_path) -> None:
    tracker = create_tracker(tmp_path)

    artifact = tmp_path / "model.txt"
    artifact.write_text("xor model")

    tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    tracker.log_artifact(run, str(artifact))

    tracker.finish_run(run)

    artifacts = mlflow.artifacts.list_artifacts(run_id=run.id)

    assert any(
        artifact_info.path == "model.txt"
        for artifact_info in artifacts
    )