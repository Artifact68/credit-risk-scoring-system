from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.ml.model_service import ModelNotReadyError
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

router = APIRouter()


def get_model_service(request: Request):
    return request.app.state.model_service


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    service = get_model_service(request)
    return HealthResponse(
        status="ok",
        model_loaded=service.loaded or service.load(),
        model_path=str(service.model_path),
    )


@router.get("/api/v1/model", tags=["model"])
def model_info(request: Request) -> dict:
    return get_model_service(request).info()


@router.get("/api/v1/schema", tags=["model"])
def model_schema(request: Request) -> dict:
    try:
        return get_model_service(request).schema()
    except ModelNotReadyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post("/api/v1/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    try:
        result = get_model_service(request).predict(payload.features)
        return PredictionResponse(**result)
    except ModelNotReadyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process input values: {error}",
        ) from error
