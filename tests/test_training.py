from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.training import build_defaults, build_schema, train_model


def make_dataset(rows: int = 160) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    income = rng.normal(70_000, 18_000, rows).clip(20_000)
    debt = rng.normal(24_000, 12_000, rows).clip(0)
    missed_payments = rng.integers(0, 5, rows)
    employment = rng.choice(["employee", "self_employed", "student"], rows)

    logit = -2.0 + debt / 35_000 + missed_payments * 0.55 - income / 120_000
    probability = 1 / (1 + np.exp(-logit))
    target = pd.Series(rng.binomial(1, probability), name="default_flg")
    frame = pd.DataFrame(
        {
            "income": income,
            "debt": debt,
            "missed_payments": missed_payments,
            "employment": employment,
        }
    )
    return frame, target


def test_trains_pipeline_and_returns_metrics():
    frame, target = make_dataset()
    pipeline, metrics, threshold = train_model(frame, target)

    probabilities = pipeline.predict_proba(frame.head(3))[:, 1]

    assert probabilities.shape == (3,)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 < threshold < 1.0


def test_builds_frontend_schema_with_defaults():
    frame, _ = make_dataset(20)
    defaults = build_defaults(frame)
    schema = build_schema(frame, defaults)

    by_name = {item["name"]: item for item in schema}
    assert by_name["income"]["type"] == "number"
    assert by_name["employment"]["type"] == "select"
    assert defaults["employment"] in by_name["employment"]["options"]
