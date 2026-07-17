from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.training import (
    load_competition_data,
    prepare_training_frame,
    save_artifact,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit scoring model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-path", type=Path, default=Path("models/credit_scoring.joblib"))
    parser.add_argument("--metrics-path", type=Path, default=Path("models/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_competition_data(args.data_dir)
    frame, target = prepare_training_frame(bundle)
    pipeline, metrics, threshold = train_model(frame, target)

    metadata = {
        "model_type": "LogisticRegression",
        "training_rows": len(frame),
        "source_files": bundle.source_files,
        "metrics": metrics,
    }
    save_artifact(
        path=args.model_path,
        pipeline=pipeline,
        frame=frame,
        target_name=bundle.target,
        threshold=threshold,
        metadata=metadata,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(frame)}")
    print(f"Features: {frame.shape[1]}")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"Average precision: {metrics['average_precision']}")
    print(f"F1: {metrics['f1']}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Model saved to: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
