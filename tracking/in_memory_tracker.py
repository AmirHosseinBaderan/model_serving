from datetime import datetime
from uuid import uuid4

from .run import Run,RunStatus,MetricValue
from .tracker import ExperimentTracker
from .experiment import Experiment

class InMemoryExperimentTracker(ExperimentTracker):
    def __init__(self):
        self.runs: dict[str,Run] = {}
        self.experiments: dict[str, Experiment] = {}
        
    def start_run(self, experiment_name)-> Run:       
        run = Run(
            id=str(uuid4()),
            experiment_name=experiment_name,
            started_at=datetime.now()
        )
        
        self.runs[run.id] = run
        
        experiment = self.experiments.get(experiment_name)

        if experiment is not None:
            experiment.runs.append(run.id)
        
        return run
    
    def log_parameter(self, run, name, value):
        run.ensure_active()
        
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

        if name not in run.metrics:
            run.metrics[name] = []

        run.metrics[name].append(
            MetricValue(
                value=value,
                step=step,
            )
        )
        
    def finish_run(
        self,
        run: Run,
    ) -> None:
        run.ensure_active()
        
        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.now()

    def fail_run(
        self,
        run: Run,
    ) -> None:
        run.ensure_active()
        
        run.status = RunStatus.FAILED
        run.finished_at = datetime.now()
        
        
    def log_artifact(
        self,
        run: Run,
        path: str,
    ) -> None:
        run.ensure_active()

        run.artifacts.append(path)
        
    def log_metadata(self, run, name, value):
        run.ensure_active()
        
        run.metadata[name] = value
        
    def create_experiment(
        self,
        name: str,
    ) -> Experiment:
        if name in self.experiments:
            raise ValueError("Experiment already exists")
    
        experiment = Experiment(name=name)
    
        self.experiments[name] = experiment
    
        return experiment
    
    def get_experiment(self, name):
        if name not in self.experiments:
            raise ValueError("Experiment not found")
        
        return self.experiments[name]
    
    def get_experiment_runs(self, experiment_name)-> list[Run]:
        experiment = self.get_experiment(experiment_name)
        
        return [
            self.runs[run_id]
            for run_id in experiment.runs
        ]
        
    def get_run(self, run_id)-> Run:
        run = self.runs.get(run_id)
        
        if run is None:
            raise ValueError("Run not found")
        
        return run