from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


class StubModelService:
    loaded = True
    model_path = "models/test.joblib"

    def load(self):
        return True

    def info(self):
        return {"loaded": True, "model_type": "test"}

    def schema(self):
        return {
            "features": [
                {
                    "name": "income",
                    "label": "Income",
                    "type": "number",
                    "required": False,
                    "default": 50_000,
                    "min": 0,
                    "max": 1_000_000,
                }
            ],
            "target": "default_flg",
            "threshold": 0.5,
        }

    def predict(self, values):
        assert values["income"] == 70_000
        return {
            "default_probability": 0.18,
            "score": 820,
            "risk_level": "low",
            "decision": "approve",
            "threshold": 0.5,
            "top_factors": [],
        }


def test_health_endpoint():
    with TestClient(app) as client:
        app.state.model_service = StubModelService()
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    with TestClient(app) as client:
        app.state.model_service = StubModelService()
        response = client.post(
            "/api/v1/predict",
            json={"features": {"income": 70_000}},
        )

    assert response.status_code == 200
    assert response.json()["score"] == 820
    assert response.json()["decision"] == "approve"
