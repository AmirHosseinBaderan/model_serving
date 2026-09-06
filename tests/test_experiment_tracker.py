from tracking.in_memory_tracker import InMemoryExperimentTracker
from tracking.run import RunStatus,MetricValue

import pytest

def test_start_run_creates_run() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    assert run.experiment_name == "xor"
    assert run.id
    assert run.started_at is not None


def test_log_parameter_stores_parameter() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_parameter(
        run,
        "learning_rate",
        0.01,
    )

    assert run.parameters["learning_rate"] == 0.01


def test_log_metric_stores_metric_history() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_metric(
        run,
        "loss",
        0.8,
        step=1
    )

    tracker.log_metric(
        run,
        "loss",
        0.4,
        step=2
    )

    assert run.metrics["loss"] == [
        MetricValue(value=0.8, step=1),
        MetricValue(value=0.4, step=2),
    ]
    
    
def test_new_run_is_running() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    assert run.status == RunStatus.RUNNING


def test_finish_run_marks_run_as_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.finish_run(run)

    assert run.status == RunStatus.COMPLETED


def test_fail_run_marks_run_as_failed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.fail_run(run)

    assert run.status == RunStatus.FAILED
    
def test_cannot_log_metric_after_run_is_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_metric(
            run,
            "loss",
            0.1,
            step=1
        )


def test_cannot_finish_completed_run() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.finish_run(run)


def test_cannot_fail_completed_run() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.fail_run(run)
        
def test_cannot_log_metric_after_run_has_failed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.fail_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_metric(
            run,
            "loss",
            0.1,
            step=1,
        )
        
def test_cannot_log_parameter_after_run_is_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_parameter(
            run,
            "learning_rate",
            0.001,
        )


def test_cannot_log_parameter_after_run_has_failed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.fail_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_parameter(
            run,
            "learning_rate",
            0.001,
        )
        
def test_log_artifact_stores_artifact() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_artifact(
        run,
        "model.pt",
    )

    assert run.artifacts == ["model.pt"]
    
def test_cannot_log_artifact_after_run_is_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_artifact(
            run,
            "model.pt",
        )
        
def test_log_metadata_stores_metadata() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_metadata(
        run,
        "git_commit",
        "a83f21c",
    )

    assert run.metadata["git_commit"] == "a83f21c"
    
def test_cannot_log_metadata_after_run_is_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_metadata(
            run,
            "git_commit",
            "a83f21c",
        )
        
def test_log_environment_metadata() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_metadata(
        run,
        "python_version",
        "3.11.13",
    )

    tracker.log_metadata(
        run,
        "pytorch_version",
        "2.7.1",
    )

    assert run.metadata["python_version"] == "3.11.13"
    assert run.metadata["pytorch_version"] == "2.7.1"
    
def test_log_code_version() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_metadata(
        run,
        "git_commit",
        "a83f21c",
    )

    assert run.metadata["git_commit"] == "a83f21c"
    
def test_cannot_log_code_version_after_run_is_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_metadata(
            run,
            "git_commit",
            "a83f21c",
        )
        
def test_log_dataset_version() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_metadata(
        run,
        "dataset_version",
        "v1",
    )

    assert run.metadata["dataset_version"] == "v1"
    
def test_cannot_log_dataset_version_after_run_is_completed() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")
    tracker.finish_run(run)

    with pytest.raises(
        ValueError,
        match="Run is not active",
    ):
        tracker.log_metadata(
            run,
            "dataset_version",
            "v1",
        )
        
def test_log_multiple_parameters() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_parameter(
        run,
        "learning_rate",
        0.001,
    )

    tracker.log_parameter(
        run,
        "epochs",
        100,
    )

    tracker.log_parameter(
        run,
        "optimizer",
        "Adam",
    )

    assert run.parameters["learning_rate"] == 0.001
    assert run.parameters["epochs"] == 100
    assert run.parameters["optimizer"] == "Adam"
    
    
def test_log_metric_stores_step() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    tracker.log_metric(
        run,
        "train_loss",
        0.42,
        step=1,
    )

    assert run.metrics["train_loss"][0] == MetricValue(
        value=0.42,
        step=1,
    )
    
def test_log_metric_rejects_negative_step() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    with pytest.raises(
        ValueError,
        match="Metric step cannot be negative",
    ):
        tracker.log_metric(
            run,
            "train_loss",
            0.42,
            step=-1,
        )
        
def test_create_experiment() -> None:
    tracker = InMemoryExperimentTracker()

    experiment = tracker.create_experiment("xor")

    assert experiment.name == "xor"
    
def test_create_duplicate_experiment_fails() -> None:
    tracker = InMemoryExperimentTracker()

    tracker.create_experiment("xor")

    with pytest.raises(
        ValueError,
        match="Experiment already exists",
    ):
        tracker.create_experiment("xor")
        
def test_get_experiment() -> None:
    tracker = InMemoryExperimentTracker()

    created = tracker.create_experiment("xor")

    experiment = tracker.get_experiment("xor")

    assert experiment is created
    
def test_get_unknown_experiment_fails() -> None:
    tracker = InMemoryExperimentTracker()

    with pytest.raises(
        ValueError,
        match="Experiment not found",
    ):
        tracker.get_experiment("unknown")
        
def test_experiment_contains_runs() -> None:
    tracker = InMemoryExperimentTracker()

    experiment = tracker.create_experiment("xor")
    run = tracker.start_run("xor")

    assert run.id in experiment.runs
    
def test_get_experiment_runs() -> None:
    tracker = InMemoryExperimentTracker()

    tracker.create_experiment("xor")

    run1 = tracker.start_run("xor")
    run2 = tracker.start_run("xor")

    runs = tracker.get_experiment_runs("xor")

    assert runs == [run1, run2]
    
def test_get_runs_for_unknown_experiment_fails() -> None:
    tracker = InMemoryExperimentTracker()

    with pytest.raises(
        ValueError,
        match="Experiment not found",
    ):
        tracker.get_experiment_runs("unknown")
        
def test_get_run() -> None:
    tracker = InMemoryExperimentTracker()

    run = tracker.start_run("xor")

    result = tracker.get_run(run.id)

    assert result is run
    
def test_get_unknown_run_fails() -> None:
    tracker = InMemoryExperimentTracker()

    with pytest.raises(
        ValueError,
        match="Run not found",
    ):
        tracker.get_run("unknown")