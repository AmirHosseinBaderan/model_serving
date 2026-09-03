from pathlib import Path
from .model_service import ModelService

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.pt"
)

class ModelServer:
    def __init__(self,model_path:str | Path):
        self.service = ModelService(model_path)
        self._running = False
        
    @property
    def is_running(self)-> bool:
        return self._running
    
    @property
    def is_ready(self)-> bool:
        return self.service.is_ready
    
    def start(self):
        if self._running:
            return
        
        self.service.start()
        self._running = True
        
    def stop(self):
        if not self._running:
            return
        
        self._running = False
    
    def predict(self,inputs:list[list[float]]) -> list[float]:
        if not self.is_running:
            raise RuntimeError(
                "model server is not running."
            )
            
        return self.service.predict(inputs)
    
def create_server()-> ModelServer:
    return ModelServer(MODEL_PATH)

if __name__ == "__main__":
    server = create_server()
    server.start()
    
    print(f"Server running : {server.is_running}")
    print(f"Model ready : {server.is_ready}")
    
    result = server.predict(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    
    print(f"Predictions : {result}")
    
    server.stop()
    
    print(f"Server running : {server.is_running}")