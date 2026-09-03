from pathlib import Path
from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager

from .schemas import (
    PredictionRequest,
    PredictionResponse
)
from serving.server import ModelServer

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "artifacts"
    / "model.pt"
)

server = ModelServer(MODEL_PATH)

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("fastapi app started")
    server.start()
    
    yield
    print("fastapi app stoped")
    server.stop()
    
app = FastAPI(
    title="Production AI",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {
        "ready": server.is_ready,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:

    if not server.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    predictions = server.predict(request.inputs)

    return PredictionResponse(
        predictions=predictions,
    )