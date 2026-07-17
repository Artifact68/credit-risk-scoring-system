from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = (
    "default_flg",
    "default_flag",
    "target",
    "default",
    "label",
    "bad_flag",
)
ID_CANDIDATES = (
    "application_id",
    "app_id",
    "client_id",
    "customer_id",
    "credit_id",
    "id",
)


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    target: str
    id_columns: list[str]
    source_files: list[str]


def read_csv(path: Path) -> pd.DataFrame:
    attempts = (
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "cp1251"},
    )
    last_error: Exception | None = None
    for options in attempts:
        try:
            return pd.read_csv(path, low_memory=False, **options)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Could not read {path}: {last_error}")


def find_csv_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def load_competition_data(data_dir: Path) -> DatasetBundle:
    files = find_csv_files(data_dir)
    tables = {path.name: read_csv(path) for path in files}

    direct = _find_table_with_target(tables)
    if direct is not None:
        name, frame, target = direct
        ids = _detect_id_columns(frame.columns)
        return DatasetBundle(frame=frame, target=target, id_columns=ids, source_files=[name])

    target_item = _find_target_table(tables)
    feature_item = _find_feature_table(
        tables, excluded_name=target_item[0] if target_item else None
    )
    if target_item is None or feature_item is None:
        available = ", ".join(tables)
        raise ValueError(
            "Could not identify feature and target files. "
            f"Available CSV files: {available}"
        )

    target_name, target_frame, target = target_item
    feature_name, feature_frame = feature_item
    common_ids = _common_id_columns(feature_frame.columns, target_frame.columns)
    if not common_ids:
        raise ValueError(
            f"No common identifier found between {feature_name} and {target_name}."
        )

    merged = feature_frame.merge(
        target_frame[common_ids + [target]],
        on=common_ids,
        how="inner",
        validate="one_to_one",
    )
    return DatasetBundle(
        frame=merged,
        target=target,
        id_columns=common_ids,
        source_files=[feature_name, target_name],
    )


def _find_table_with_target(
    tables: dict[str, pd.DataFrame],
) -> tuple[str, pd.DataFrame, str] | None:
    candidates: list[tuple[str, pd.DataFrame, str]] = []
    for name, frame in tables.items():
        target = _detect_target(frame.columns)
        if target and frame.shape[1] > 2:
            candidates.append((name, frame, target))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1].shape[1])


def _find_target_table(
    tables: dict[str, pd.DataFrame],
) -> tuple[str, pd.DataFrame, str] | None:
    candidates: list[tuple[str, pd.DataFrame, str]] = []
    for name, frame in tables.items():
        target = _detect_target(frame.columns)
        if target:
            candidates.append((name, frame, target))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1].shape[1])


def _find_feature_table(
    tables: dict[str, pd.DataFrame], excluded_name: str | None
) -> tuple[str, pd.DataFrame] | None:
    candidates = [
        (name, frame)
        for name, frame in tables.items()
        if name != excluded_name and frame.shape[1] > 2
    ]
    if not candidates:
        return None

    preferred = [item for item in candidates if "application" in item[0].lower()]
    return max(preferred or candidates, key=lambda item: item[1].shape[1])


def _detect_target(columns: Iterable[str]) -> str | None:
    normalized = {str(column).lower(): str(column) for column in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _detect_id_columns(columns: Iterable[str]) -> list[str]:
    normalized = {str(column).lower(): str(column) for column in columns}
    return [normalized[name] for name in ID_CANDIDATES if name in normalized]


def _common_id_columns(left: Iterable[str], right: Iterable[str]) -> list[str]:
    left_map = {str(column).lower(): str(column) for column in left}
    right_names = {str(column).lower() for column in right}
    for candidate in ID_CANDIDATES:
        if candidate in left_map and candidate in right_names:
            return [left_map[candidate]]

    common = [column for column in left if str(column) in set(map(str, right))]
    return common[:1]


def prepare_training_frame(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.Series]:
    frame = bundle.frame.copy()
    frame = frame.dropna(subset=[bundle.target])

    target = pd.to_numeric(frame.pop(bundle.target), errors="coerce")
    valid_mask = target.notna()
    frame = frame.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].astype(int).reset_index(drop=True)

    for column in bundle.id_columns:
        if column in frame:
            frame = frame.drop(columns=column)

    frame = _normalize_values(frame)
    frame = _drop_unusable_columns(frame)

    unique_target = sorted(target.unique().tolist())
    if unique_target != [0, 1]:
        raise ValueError(f"Target must contain 0 and 1, got {unique_target}")
    if frame.empty:
        raise ValueError("No usable features remain after preprocessing")

    return frame, target


def _normalize_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].astype("string")
        elif result[column].dtype == object:
            result[column] = result[column].replace(r"^\s*$", np.nan, regex=True)
            numeric = pd.to_numeric(result[column], errors="coerce")
            if numeric.notna().mean() > 0.95:
                result[column] = numeric
            else:
                result[column] = result[column].astype("string")
    return result


def _drop_unusable_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if series.isna().mean() >= 0.98:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        if series.dtype == object and series.nunique(dropna=True) > len(series) * 0.9:
            continue
        keep.append(column)
    return frame[keep]


def build_pipeline(frame: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    categorical = frame.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline, numeric, categorical


def train_model(
    frame: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], float]:
    x_train, x_test, y_train, y_test = train_test_split(
        frame,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    pipeline, _, _ = build_pipeline(frame)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "threshold": round(threshold, 4),
        "test_rows": int(len(y_test)),
        "positive_rate": round(float(target.mean()), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }
    return pipeline, metrics, threshold


def choose_threshold(target: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def build_defaults(frame: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            value = series.median()
            defaults[column] = None if pd.isna(value) else float(value)
        else:
            mode = series.mode(dropna=True)
            defaults[column] = None if mode.empty else str(mode.iloc[0])
    return defaults


def build_schema(frame: pd.DataFrame, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        item: dict[str, Any] = {
            "name": column,
            "label": column.replace("_", " ").strip().title(),
            "required": False,
            "default": defaults.get(column),
        }
        if pd.api.types.is_numeric_dtype(series):
            item["type"] = "number"
            minimum = series.min(skipna=True)
            maximum = series.max(skipna=True)
            item["min"] = None if pd.isna(minimum) else float(minimum)
            item["max"] = None if pd.isna(maximum) else float(maximum)
        else:
            values = series.dropna().astype(str).value_counts().head(30).index.tolist()
            item["type"] = "select" if len(values) <= 30 else "text"
            item["options"] = values if item["type"] == "select" else []
        schema.append(item)
    return schema


def save_artifact(
    path: Path,
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target_name: str,
    threshold: float,
    metadata: dict[str, Any],
) -> None:
    defaults = build_defaults(frame)
    artifact = {
        "pipeline": pipeline,
        "features": frame.columns.tolist(),
        "target": target_name,
        "threshold": threshold,
        "defaults": defaults,
        "schema": build_schema(frame, defaults),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
