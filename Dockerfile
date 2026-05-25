FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
        torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "voiceguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
