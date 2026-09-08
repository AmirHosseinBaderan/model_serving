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

        self.runs: dict[str, Run] = {}

    def create_experiment(self, name: str) -> Experiment:
        if mlflow.get_experiment_by_name(name) is not None:
            raise ValueError("Experiment already exists")

        mlflow.create_experiment(name)

        return Experiment(name=name)

    def get_experiment(self, name: str) -> Experiment:
        mlflow_experiment = mlflow.get_experiment_by_name(name)

        if mlflow_experiment is None:
            raise ValueError("Experiment not found")

        experiment = Experiment(name=name)

        for run in mlflow.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id],
            output_format="list",
        ):
            experiment.runs.append(run.info.run_id)

        return experiment

    def start_run(self, experiment_name: str) -> Run:
        experiment = self.get_experiment(experiment_name)
    
        if mlflow.active_run() is not None:
            raise RuntimeError("Another MLflow run is already active")
    
        mlflow_experiment = mlflow.get_experiment_by_name(
            experiment_name
        )
    
        mlflow_run = mlflow.start_run(
            experiment_id=mlflow_experiment.experiment_id
        )
    
        run = Run(
            id=mlflow_run.info.run_id,
            experiment_name=experiment.name,
            started_at=datetime.now(),
        )
    
        self.runs[run.id] = run
    
        return run

    def get_run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)

        if run is not None:
            return run

        mlflow_run = mlflow.get_run(run_id)

        run = Run(
            id=run_id,
            experiment_name=mlflow_run.data.tags.get(
                "mlflow.experimentName",
                "",
            ),
            started_at=datetime.fromtimestamp(
                mlflow_run.info.start_time / 1000
            ),
        )

        run.status = RunStatus.COMPLETED

        self.runs[run.id] = run

        return run

    def get_experiment_runs(
        self,
        experiment_name: str,
    ) -> list[Run]:
        experiment = self.get_experiment(experiment_name)

        return [
            self.get_run(run_id)
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
        
    def _ensure_no_active_run(self) -> None:
        if mlflow.active_run() is not None:
            raise RuntimeError("Another MLflow run is already active")