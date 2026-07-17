from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Credit Risk Scoring System"
    app_version: str = "0.1.0"
    model_path: Path = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "credit_scoring.joblib"))
    frontend_dir: Path = Path(os.getenv("FRONTEND_DIR", BASE_DIR / "frontend"))
    data_dir: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "raw"))


settings = Settings()
