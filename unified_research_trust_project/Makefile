.PHONY: install pipeline train explain validate app openalex trust-ebm clean

PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements.txt

pipeline: train explain

openalex:
	$(BIN)/python src/download_public_reference.py --per-page 80

data:
	$(BIN)/python src/generate_data.py --n-samples 1500

train: data
	$(BIN)/python src/train_models.py

explain:
	$(BIN)/python src/explain_and_export.py

validate:
	$(BIN)/python src/validate_project.py

app:
	$(BIN)/streamlit run app/streamlit_app.py

trust-ebm:
	$(BIN)/python src/train_model_ebm_trust.py

clean:
	rm -rf $(VENV) __pycache__ app/__pycache__ src/__pycache__
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
