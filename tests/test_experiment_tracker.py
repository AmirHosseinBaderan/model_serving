from tracking.in_memory_tracker import InMemoryExperimentTracker
from tracking.run import RunStatus

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
    )

    tracker.log_metric(
        run,
        "loss",
        0.4,
    )

    assert run.metrics["loss"] == [0.8, 0.4]
    
    
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