from dataclasses import dataclass,field
from datetime import datetime
from typing import Any

from enum import Enum

class RunStatus(str,Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Run:
    id: str
    experiment_name: str
    started_at: datetime
    status:RunStatus = RunStatus.RUNNING
    
    parameters: dict[str,Any] = field(
        default_factory=dict
    )
    
    metrics: dict[str,list[float]] = field(
        default_factory=dict
    )
    
    artifacts: list[str] = field(
        default_factory=list
    )
    
    def ensure_active(self) -> None:
        if self.status != RunStatus.RUNNING:
            raise ValueError("Run is not active")