from dataclasses import dataclass,field
from datetime import datetime
from typing import Any

@dataclass
class Run:
    id: str
    experiment_name: str
    started_at: datetime
    
    parameters: dict[str,Any] = field(
        default_factory=dict
    )
    
    metrics: dict[str,list[float]] = field(
        default_factory=dict
    )