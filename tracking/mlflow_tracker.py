from datetime import datetime
from typing import Any

import mlflow

from tracking.experiment import Experiment
from tracking.run import MetricValue, Run, RunStatus
from tracking.tracker import ExperimentTracker


class MLflowExperimentTracker(ExperimentTracker):
    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiments: dict[str, Experiment] = {}
        self.runs: dict[str, Run] = {}

    def create_experiment(self, name: str) -> Experiment:
        if name in self.experiments:
            raise ValueError("Experiment already exists")

        experiment_id = mlflow.create_experiment(name)

        experiment = Experiment(name=name)
        self.experiments[name] = experiment

        return experiment

    def get_experiment(self, name: str) -> Experiment:
        experiment = self.experiments.get(name)

        if experiment is None:
            raise ValueError("Experiment not found")

        return experiment

    def start_run(self, experiment_name: str) -> Run:
        experiment = self.get_experiment(experiment_name)

        mlflow_run = mlflow.start_run(
            experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id
        )

        run = Run(
            id=mlflow_run.info.run_id,
            experiment_name=experiment.name,
            started_at=datetime.now(),
        )

        self.runs[run.id] = run
        experiment.runs.append(run.id)

        return run

    def get_run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)

        if run is None:
            raise ValueError("Run not found")

        return run

    def get_experiment_runs(self, experiment_name: str) -> list[Run]:
        experiment = self.get_experiment(experiment_name)

        return [
            self.runs[run_id]
            for run_id in experiment.runs
        ]

    def log_parameter(
        self,
        run: Run,
        name: str,
        value: Any,
    ) -> None:
        run.ensure_active()

        mlflow.log_param(name, value)
        run.parameters[name] = value

    def log_metric(
        self,
        run: Run,
        name: str,
        value: float,
        step: int,
    ) -> None:
        run.ensure_active()

        if step < 0:
            raise ValueError("Metric step cannot be negative")

        mlflow.log_metric(
            name,
            value,
            step=step,
        )

        if name not in run.metrics:
            run.metrics[name] = []

        run.metrics[name].append(
            MetricValue(
                value=value,
                step=step,
            )
        )

    def log_metadata(
        self,
        run: Run,
        name: str,
        value: Any,
    ) -> None:
        run.ensure_active()

        mlflow.set_tag(name, value)
        run.metadata[name] = value

    def log_artifact(
        self,
        run: Run,
        path: str,
    ) -> None:
        run.ensure_active()

        if not path:
            raise ValueError("Artifact path cannot be empty")

        mlflow.log_artifact(path)
        run.artifacts.append(path)

    def finish_run(self, run: Run) -> None:
        run.ensure_active()

        mlflow.end_run()

        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.now()

    def fail_run(self, run: Run) -> None:
        run.ensure_active()

        mlflow.end_run(status="FAILED")

        run.status = RunStatus.FAILED
        run.finished_at = datetime.now()

    def get_run_summary(
        self,
        run_id: str,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)

        return {
            "id": run.id,
            "experiment_name": run.experiment_name,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "duration": run.duration,
            "status": run.status,
            "parameters": run.parameters,
            "metrics": run.metrics,
            "metadata": run.metadata,
            "artifacts": run.artifacts,
        }