from tracking.in_memory_tracker import InMemoryExperimentTracker


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