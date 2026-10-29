FROM python:3.11-slim AS base
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

FROM base AS runtime
COPY agent/ agent/
COPY configs/ configs/
COPY deploy/inference_server.py inference_server.py
EXPOSE 8000
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
