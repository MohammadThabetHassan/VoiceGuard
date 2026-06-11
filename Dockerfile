FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# CPU torch wheels via --extra-index-url in the SAME resolve (mirrors CI):
# installing from PyPI first would pull the ~2.5 GB CUDA torch and then
# downgrade, and a pinned torch 2.1.x predates NumPy 2.x ABI support — it
# crashes on import against this project's numpy>=2.4.6.
RUN pip install --no-cache-dir -e . \
    --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

# --proxy-headers + --forwarded-allow-ips "*": behind the compose nginx, trust
# X-Forwarded-For so rate limiting keys on the real client IP (container network).
CMD ["uvicorn", "voiceguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
