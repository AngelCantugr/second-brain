# Test container: its only purpose is to run the real-backend integration
# tests + benchmarks in integration/ against a live Ollama service, and
# write the results to /artifacts.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY second_brain ./second_brain
COPY tests ./tests
COPY integration ./integration
COPY README.md ./

RUN uv sync --dev --frozen

RUN chmod +x integration/entrypoint.sh

ENTRYPOINT ["integration/entrypoint.sh"]
