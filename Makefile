.PHONY: setup data features train evaluate all test test-ci clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python -m src.data.convert_to_parquet
	.venv/bin/python -m src.data.validate

features:
	.venv/bin/python -m src.features.feature_engineering

train:
	.venv/bin/python -m src.models.train

evaluate:
	.venv/bin/python -m src.decision.optimal_thresholds

all: setup data features train evaluate

test:
	pytest tests/

test-ci:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache src/*/__pycache__
