from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

COMPETITION = "fintech-credit-scoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the competition dataset from Kaggle")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def copy_dataset(source: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    downloaded_path = Path(kagglehub.competition_download(COMPETITION))
    copied = copy_dataset(downloaded_path, args.output)
    print(f"Downloaded to: {downloaded_path}")
    print(f"Copied {copied} files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
