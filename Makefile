.PHONY: setup data features train evaluate all test test-ci clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python -m src.data.convert_to_parquet
	.venv/bin/python -m src.data.validate

features:
	# To be implemented in Phase 1
	echo "Running feature engineering..."

train:
	# To be implemented in Phase 2 & 3
	echo "Training models..."

evaluate:
	# To be implemented in Phase 4 & 5
	echo "Evaluating models..."

all: setup data features train evaluate

test:
	pytest tests/

test-ci:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache src/*/__pycache__
