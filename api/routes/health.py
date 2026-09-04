from fastapi import APIRouter

from api.schemas.error import ErrorResponse
from serving.server import ModelServer
from serving.exceptions import ModelNotReadyError


def create_health_router(server: ModelServer) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get(
        "/ready",
        responses={
            503: {
                "model": ErrorResponse,
            }
        },
    )
    def ready() -> dict[str, bool]:
        if not server.is_ready:
            raise ModelNotReadyError("Model is not ready.")

        return {"ready": True}

    return router