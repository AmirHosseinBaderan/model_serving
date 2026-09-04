from fastapi import APIRouter

from api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from serving.server import ModelServer


def create_prediction_router(
    server: ModelServer,
) -> APIRouter:

    router = APIRouter()

    @router.post(
        "/predict",
        response_model=PredictionResponse,
    )
    def predict(
        request: PredictionRequest,
    ) -> PredictionResponse:

        predictions = server.predict(
            request.inputs,
        )

        return PredictionResponse(
            predictions=predictions,
        )

    return router