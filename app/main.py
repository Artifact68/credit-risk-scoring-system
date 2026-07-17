from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.ml.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService(settings.model_path)
    service.load()
    app.state.model_service = service
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for estimating the probability of credit default.",
    lifespan=lifespan,
)
app.include_router(router)

if settings.frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")
