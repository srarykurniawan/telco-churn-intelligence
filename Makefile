PY ?= python
export PYTHONPATH := src

.PHONY: install install-dev data train monitor retrain predict test lint app docker-build docker-run clean-models

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

data:            ## regenerate synthetic data
	$(PY) -c "from telco.config import load_config; from telco.pipelines.train import ensure_data; ensure_data(load_config(), regenerate=True)"

train:           ## train all models & register them
	$(PY) -m telco.pipelines.train

monitor:         ## simulate a drifted batch & run monitoring (SCENARIO=none|mild|severe)
	$(PY) -m telco.pipelines.monitor --scenario $(or $(SCENARIO),mild)

retrain:         ## monitor and retrain automatically if the trigger fires
	$(PY) -m telco.pipelines.monitor --scenario $(or $(SCENARIO),severe) --auto-retrain

predict:         ## batch scoring: make predict IN=file.csv OUT=scored.csv
	$(PY) -m telco.pipelines.predict --input $(IN) --output $(OUT)

test:
	pytest --cov=telco --cov-report=term-missing

lint:
	ruff check src app tests

app:
	streamlit run app/Home.py

docker-build:
	docker build -t telco-intelligence .

docker-run:
	docker run --rm -p 8501:8501 telco-intelligence

clean-models:    ## wipe registry, data versions & monitoring (fresh start)
	rm -rf models/registry data/versions monitoring reports data/incoming/*.csv
