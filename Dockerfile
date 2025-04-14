# Base image
FROM python:3.12-slim-bookworm AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create a non-root user
RUN adduser --gecos "FastAPI User" --disabled-password --quiet "fastapi-user" --uid 1001 --gid 0 --home /opt/fastapi-app/

# Set working directory
WORKDIR /opt/fastapi-app

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Criar diretórios de dados necessários
RUN mkdir -p /opt/fastapi-app/data/chroma_db \
    && chown -R fastapi-user:0 /opt/fastapi-app/data \
    && chmod -R 755 /opt/fastapi-app/data

# Switch to non-root user
USER fastapi-user

# Set PATH
ENV PATH=/usr/local/bin:/opt/fastapi-app/.local/bin:/opt/fastapi-app/.venv/bin:$PATH

# Copy requirements file
COPY --chown=fastapi-user:0 requirements.txt /opt/fastapi-app/

# Install dependencies
RUN python -m venv /opt/fastapi-app/.venv \
    && . /opt/fastapi-app/.venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /opt/fastapi-app/requirements.txt

# Application image
FROM base AS app

# Copy application files
COPY --chown=fastapi-user:0 ./app/ /opt/fastapi-app/app/

# Copiar dados existentes (se houver)
COPY --chown=fastapi-user:0 ./data/ /opt/fastapi-app/data/

# Expose port
EXPOSE 8000

# Set entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
