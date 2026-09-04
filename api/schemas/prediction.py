from pydantic import BaseModel,Field

class PredictionRequest(BaseModel):
    inputs:list[list[float]] = Field(min_length=1)
    
class PredictionResponse(BaseModel):
    predictions:list[float]