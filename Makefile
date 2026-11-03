PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help setup data features train calibrate decide bootstrap power monitor explain tableau demo score-demo docs verify-docs test verify pipeline all clean

help:
	@echo "Breakpoint targets"
	@echo ""
	@echo "  make setup        Create .venv and install pinned requirements"
	@echo "  make data         Convert raw Kaggle CSVs to parquet and validate"
	@echo "  make features     Build customer-level features with DuckDB"
	@echo "  make train        Train LightGBM + LR baseline and write OOF predictions"
	@echo "  make calibrate    Cross-fit isotonic/Platt calibration"
	@echo "  make decide       Fit binary minimax + three-action policy"
	@echo "  make bootstrap    Paired profit bootstrap"
	@echo "  make power        Champion/challenger power analysis"
	@echo "  make monitor      Customer-level PSI and action-mix monitoring"
	@echo "  make explain      SHAP on the saved model + reason codes"
	@echo "  make tableau      Export the 12 Tableau aggregate CSVs"
	@echo "  make demo         Build docs/index.html from the reports"
	@echo "  make score-demo   Score data/demo/scoring_input.csv"
	@echo "  make docs         Render generated docs blocks"
	@echo "  make verify-docs  Fail if docs drift from reports"
	@echo "  make test         Run the test suite"
	@echo "  make verify       Tests + docs check + demo rebuild"
	@echo "  make pipeline     Full pipeline (requires data/raw/*.csv)"
	@echo "  make clean        Remove caches and generated demo outputs"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PY) -m src.data.convert_to_parquet
	$(PY) -m src.data.validate

features:
	$(PY) -m src.features.feature_engineering

train:
	$(PY) -m src.models.train

calibrate:
	$(PY) -m src.models.calibration

decide:
	$(PY) -m src.decision.optimal_thresholds

bootstrap:
	$(PY) -m src.evaluation.bootstrap

power:
	$(PY) -m src.evaluation.power_analysis

monitor:
	$(PY) -m src.monitoring.psi

explain:
	$(PY) -m src.explainability.shap_analysis

tableau:
	$(PY) -m src.decision.export_dashboard_data

demo:
	$(PY) -m src.demo.build_site

score-demo:
	$(PY) -m src.scoring.score --input data/demo/scoring_input.csv --output reports/scored_demo.csv --explain --explain-rows 20

docs:
	$(PY) scripts/render_docs.py --write

verify-docs:
	$(PY) scripts/verify_docs.py

test:
	$(PY) -m pytest tests/ -q

verify: test verify-docs demo
	@echo "Verification passed."

pipeline: data features train calibrate decide bootstrap power monitor explain tableau demo docs
	$(PY) scripts/verify_docs.py

all: pipeline

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
