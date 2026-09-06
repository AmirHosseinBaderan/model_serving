from typing import Any
from .run import Run

class ExperimentTracker:
    def start_run(
        self,
        experiment_name:str
    )-> Run:
        raise NotImplementedError
    
    def log_parameter(
        self,
        run:Run,
        name: str,
        value: Any
    ):
        raise NotImplementedError
    
    def log_metric(
        self,
        run: Run,
        name: str,
        value: float
    ):
        raise NotImplementedError
    
    def finish_run(
        self,
        run: Run,
    ) -> None:
        raise NotImplementedError

    def fail_run(
        self,
        run: Run,
    ) -> None:
        raise NotImplementedError
    
    def log_artifact(
        self,
        run: Run,
        path: str,
    ) -> None:
        raise NotImplementedError