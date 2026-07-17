from __future__ import annotations

import pandas as pd

from app.ml.training import load_competition_data, prepare_training_frame


def test_loads_and_merges_separate_feature_and_target_files(tmp_path):
    applications = pd.DataFrame(
        {
            "app_id": [1, 2, 3, 4],
            "income": [60_000, 45_000, 90_000, 38_000],
            "employment": ["office", "service", "office", "service"],
        }
    )
    targets = pd.DataFrame(
        {
            "app_id": [1, 2, 3, 4],
            "default_flg": [0, 1, 0, 1],
        }
    )
    applications.to_csv(tmp_path / "application_info.csv", index=False)
    targets.to_csv(tmp_path / "default_flg.csv", index=False)

    bundle = load_competition_data(tmp_path)
    frame, target = prepare_training_frame(bundle)

    assert bundle.target == "default_flg"
    assert bundle.id_columns == ["app_id"]
    assert frame.columns.tolist() == ["income", "employment"]
    assert target.tolist() == [0, 1, 0, 1]


def test_removes_constant_and_almost_empty_columns(tmp_path):
    frame = pd.DataFrame(
        {
            "client_id": range(10),
            "income": range(10_000, 20_000, 1_000),
            "constant": "same",
            "mostly_empty": [None] * 10,
            "default_flg": [0, 1] * 5,
        }
    )
    frame.to_csv(tmp_path / "train.csv", index=False)

    bundle = load_competition_data(tmp_path)
    features, _ = prepare_training_frame(bundle)

    assert "constant" not in features
    assert "mostly_empty" not in features
    assert "client_id" not in features
