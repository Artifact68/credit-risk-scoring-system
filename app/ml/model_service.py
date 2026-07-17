from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._artifact: dict[str, Any] | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._artifact is not None

    def load(self, force: bool = False) -> bool:
        if self.loaded and not force:
            return True

        if not self.model_path.exists():
            return False

        with self._lock:
            if self.loaded and not force:
                return True
            self._artifact = joblib.load(self.model_path)
        return True

    def info(self) -> dict[str, Any]:
        if not self.loaded:
            self.load()

        if not self._artifact:
            return {
                "loaded": False,
                "model_path": str(self.model_path),
                "message": "Train the model before making predictions.",
            }

        metadata = dict(self._artifact.get("metadata", {}))
        metadata.update(
            {
                "loaded": True,
                "model_path": str(self.model_path),
                "feature_count": len(self._artifact.get("features", [])),
            }
        )
        return metadata

    def schema(self) -> dict[str, Any]:
        artifact = self._require_artifact()
        return {
            "features": artifact.get("schema", []),
            "target": artifact.get("target"),
            "threshold": artifact.get("threshold", 0.5),
        }

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        artifact = self._require_artifact()
        features: list[str] = artifact["features"]
        defaults: dict[str, Any] = artifact.get("defaults", {})

        row = {name: values.get(name, defaults.get(name)) for name in features}
        frame = pd.DataFrame([row], columns=features)

        pipeline = artifact["pipeline"]
        probability = float(pipeline.predict_proba(frame)[0, 1])
        threshold = float(artifact.get("threshold", 0.5))

        risk_level, decision = self._classify_risk(probability, threshold)
        score = int(round(1000 * (1.0 - probability)))

        return {
            "default_probability": round(probability, 6),
            "score": max(0, min(1000, score)),
            "risk_level": risk_level,
            "decision": decision,
            "threshold": threshold,
            "top_factors": self._explain(frame, pipeline),
        }

    def _require_artifact(self) -> dict[str, Any]:
        if not self.loaded:
            self.load()
        if not self._artifact:
            raise ModelNotReadyError(
                f"Model artifact was not found at {self.model_path}. Run the training script first."
            )
        return self._artifact

    @staticmethod
    def _classify_risk(probability: float, threshold: float) -> tuple[str, str]:
        review_border = max(0.2, threshold * 0.6)
        if probability < review_border:
            return "low", "approve"
        if probability < threshold:
            return "medium", "manual_review"
        return "high", "decline"

    @staticmethod
    def _explain(frame: pd.DataFrame, pipeline: Any, limit: int = 6) -> list[dict[str, Any]]:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]
            transformed = preprocessor.transform(frame)
            if hasattr(transformed, "toarray"):
                values = transformed.toarray()[0]
            else:
                values = np.asarray(transformed)[0]
            coefficients = np.asarray(classifier.coef_[0])
            names = preprocessor.get_feature_names_out()
            contributions = values * coefficients

            order = np.argsort(np.abs(contributions))[::-1][:limit]
            factors: list[dict[str, Any]] = []
            for index in order:
                contribution = float(contributions[index])
                if abs(contribution) < 1e-10:
                    continue
                factors.append(
                    {
                        "feature": ModelService._clean_feature_name(str(names[index])),
                        "contribution": round(contribution, 4),
                        "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                    }
                )
            return factors
        except (AttributeError, KeyError, ValueError, TypeError):
            return []

    @staticmethod
    def _clean_feature_name(name: str) -> str:
        name = name.replace("numeric__", "").replace("categorical__", "")
        return name.replace("_", " ").strip()
