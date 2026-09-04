from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.health import create_health_router
from api.routes.prediction import create_prediction_router
from serving.bootstrap import create_server


server = create_server()


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

app.include_router(
    create_health_router(server),
    prefix="/api/v1",
)

app.include_router(
    create_prediction_router(server),
    prefix="/api/v1",
)