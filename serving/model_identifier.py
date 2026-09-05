from dataclasses import dataclass

@dataclass(frozen=True)
class ModelIdentifier:
    name:str
    version:str
    
    def __post_init__(self):
        if not self.name:
            raise ValueError(
                "Model name cannot be empty."
            )
            
        if not self.version:
            raise ValueError(
                "Modle version cannot be empty"
            )
            
    def __str__(self)-> str:
        return f"{self.name}:{self.version}"        
            