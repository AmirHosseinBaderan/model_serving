from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.routes.health import create_health_router
from api.routes.prediction import create_prediction_router
from api.schemas.error import ErrorResponse
from serving.bootstrap import create_server
from serving.exceptions import ModelNotReadyError
from serving.server import ModelServer


def create_app(server: ModelServer | None = None) -> FastAPI:

    server = server or create_server()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("FastAPI app started")

        server.start()

        yield

        print("FastAPI app stopped")

        server.stop()

    app = FastAPI(
        title="Production AI",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(
        request,
        exc: ModelNotReadyError,
    ) -> JSONResponse:

        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error={
                    "code": "MODEL_NOT_READY",
                    "message": str(exc),
                }
            ).model_dump(),
        )

    app.include_router(
        create_health_router(server),
        prefix="/api/v1",
    )

    app.include_router(
        create_prediction_router(server),
        prefix="/api/v1",
    )

    return app


app = create_app()