import json 
from pathlib import Path
from .model_metadata import ModelMetadata

class ModelMetadataLoader:
    def load(
        self,
        metadata_path:str| Path
    )-> ModelMetadata:
        path = Path(metadata_path)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {path}"
            )
            
        with path.open(
            "r",encoding="utf-8"
        ) as file:
            data = json.load(file)
        
        return ModelMetadata(**data)