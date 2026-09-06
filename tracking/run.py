from dataclasses import dataclass,field
from datetime import datetime
from typing import Any

from enum import Enum

class RunStatus(str,Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MetricValue:
    value: float
    step: int

@dataclass
class Run:
    id: str
    experiment_name: str
    started_at: datetime
    finished_at: datetime | None = None
    status:RunStatus = RunStatus.RUNNING
    
    parameters: dict[str,Any] = field(
        default_factory=dict
    )
    
    metrics: dict[str, list[MetricValue]] = field(default_factory=dict)
    
    artifacts: list[str] = field(
        default_factory=list
    )
    
    metadata:dict[str,Any] = field(default_factory=dict)
    
    def ensure_active(self) -> None:
        if self.status != RunStatus.RUNNING:
            raise ValueError("Run is not active")