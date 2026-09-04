from pydantic import BaseModel,Field

class ErrorDetail(BaseModel):
    code:str
    message:str
    
class ErrorResponse(BaseModel):
    error:ErrorDetail