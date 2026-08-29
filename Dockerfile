# MarketPulse AI - Dockerfile (Streamlit dashboard service)
FROM python:3.11-slim

WORKDIR /app

# System dependencies (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the full data/model pipeline at build time so the dashboard has data
# and a trained model available immediately when the container starts.
RUN python src/data/generate_dataset.py \
 && python src/data/data_cleaning.py \
 && python src/features/feature_engineering.py \
 && python src/analytics/marketing_analytics.py \
 && python src/models/marketing_mix_model.py \
 && python src/models/model_evaluation.py \
 && python src/models/channel_contribution.py \
 && python src/optimization/budget_optimizer.py \
 && python src/optimization/simulator.py \
 && python src/forecasting/forecast.py \
 && python src/utils/database.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
