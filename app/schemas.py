from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Applicant features used by the scoring model",
    )


class Factor(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    default_probability: float
    score: int
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "manual_review", "decline"]
    threshold: float
    top_factors: list[Factor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
