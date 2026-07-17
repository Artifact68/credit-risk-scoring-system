.PHONY: install download train run test lint

install:
	python -m pip install -r requirements-dev.txt

download:
	python scripts/download_data.py

train:
	python scripts/train.py

run:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest

lint:
	python -m ruff check .
