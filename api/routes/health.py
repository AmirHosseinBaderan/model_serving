from fastapi import APIRouter, HTTPException

from serving.server import ModelServer


def create_health_router(
    server: ModelServer,
) -> APIRouter:

    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/ready")
    def ready() -> dict[str, bool]:

        if not server.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Model is not ready.",
            )

        return {"ready": True}

    return router