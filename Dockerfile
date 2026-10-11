FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv

WORKDIR /app

# Optional extras: build with --build-arg EXTRAS=rerank to include the cross-encoder reranker
ARG EXTRAS=""

# Use the image's Python, copy packages out of uv's cache, and put the venv on PATH
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy the lockfile first for better caching
COPY pyproject.toml uv.lock ./

# Install exactly what uv.lock pins. The reranker extra gets CPU-only PyTorch (see
# pyproject.toml) and downloads its model at build time, so the first question does not wait for it
RUN if [ "$EXTRAS" = "rerank" ]; then \
        uv sync --locked --no-dev --no-cache --extra rerank \
        && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    else \
        uv sync --locked --no-dev --no-cache; \
    fi

# Copy the rest of the application
COPY . .

# Create directories if they don't exist
RUN mkdir -p data/chroma data/documents

# Expose port for Streamlit
EXPOSE 8501

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
