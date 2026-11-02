FROM python:3.11-slim
# -> mettre la même version de Python que celle de l'entraînement (le joblib en dépend)

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api ./src/api
COPY models/yield_model.joblib ./models/yield_model.joblib

RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
