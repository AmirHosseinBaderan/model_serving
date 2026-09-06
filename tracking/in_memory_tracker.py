from datetime import datetime
from uuid import uuid4

from .run import Run,RunStatus
from .tracker import ExperimentTracker

class InMemoryExperimentTracker(ExperimentTracker):
    def __init__(self):
        self.runs: dict[str,Run] = {}
        
    def start_run(self, experiment_name)-> Run:
        run = Run(
            id=str(uuid4()),
            experiment_name=experiment_name,
            started_at=datetime.now()
        )
        
        self.runs[run.id] = run
        
        return run
    
    def log_parameter(self, run, name, value):
        run.ensure_active()
        
        run.parameters[name] = value
        
    def log_metric(self, run, name, value):
        run.ensure_active()
        
        if name not in run.metrics:
            run.metrics[name] = []
            
        run.metrics[name].append(value)
        
    def finish_run(
        self,
        run: Run,
    ) -> None:
        run.ensure_active()
        
        run.status = RunStatus.COMPLETED

    def fail_run(
        self,
        run: Run,
    ) -> None:
        run.ensure_active()
        
        run.status = RunStatus.FAILED
        
    def log_artifact(
        self,
        run: Run,
        path: str,
    ) -> None:
        run.ensure_active()
    
        run.artifacts.append(path)