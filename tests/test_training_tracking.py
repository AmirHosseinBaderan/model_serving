from tracking.in_memory_tracker import InMemoryExperimentTracker
from tracking.run import RunStatus
from model.train import train


def test_training_is_tracked(tmp_path) -> None:
    tracker = InMemoryExperimentTracker()

    model_path = tmp_path / "model.pt"

    train(
        tracker=tracker,
        experiment_name="xor",
        model_path=model_path,
        epochs=10,
    )

    experiment = tracker.get_experiment("xor")

    assert len(experiment.runs) == 1

    run = tracker.get_run(experiment.runs[0])

    assert run.status == RunStatus.COMPLETED

    assert run.parameters["learning_rate"] == 0.01
    assert run.parameters["epochs"] == 10
    assert run.parameters["optimizer"] == "Adam"

    assert run.metadata["seed"] == 42

    assert "train_loss" in run.metrics
    assert len(run.metrics["train_loss"]) == 10

    assert run.artifacts == [str(model_path)]

    assert model_path.exists()