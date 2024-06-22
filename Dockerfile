# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create necessary directories with appropriate permissions
RUN mkdir -p /model_cache /tmp/.cache/huggingface /tmp/nltk_data && \
    chmod -R 777 /model_cache /tmp/.cache/huggingface /tmp/nltk_data

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV NLTK_DATA=/tmp/nltk_data

# Copy the source code and model cache scripts
COPY . .
COPY --chmod=755 model_cache ./model_cache

# Switch to the non-privileged user
USER appuser

# Run the script to download and cache the model and tokenizer
RUN ./model_cache/download_model.sh

# Expose the port that the application listens on
EXPOSE 8001

# Run the application
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8001"]