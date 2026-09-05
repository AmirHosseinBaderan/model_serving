from pathlib import Path
from serving.model_identifier import ModelIdentifier

class ModelArtifactResolver:
    def __init__(self,artifacts_root:str|Path):
        self.artifcats_root = Path(artifacts_root)
        
    def resolve(self,identifier:ModelIdentifier)-> Path:
        model_path = (
            self.artifcats_root
            / identifier.name
            / identifier.version
            / "model.onnx"
        )
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {identifier}"
            )
            
        return model_path