FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train at build time if artifacts are not committed (no-op when registry exists)
RUN python -c "import sys; sys.path.insert(0,'src'); \
from telco.config import load_config; from telco.mlops.registry import ModelRegistry; \
from telco.pipelines.train import run_training; cfg=load_config(); \
ModelRegistry(cfg.path('registry_dir')).get_production('churn_model') or run_training(cfg)"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
