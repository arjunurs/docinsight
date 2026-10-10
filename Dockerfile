FROM python:3.12-slim

WORKDIR /app

# Optional extras: build with --build-arg EXTRAS=rerank to include the cross-encoder reranker
ARG EXTRAS=""

# Copy requirements first for better caching
COPY requirements.txt requirements-rerank.txt ./

# Install dependencies. The reranker extra uses CPU-only PyTorch to keep the image small
# and downloads its model at build time, so the first question does not wait for it
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$EXTRAS" = "rerank" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
        && pip install --no-cache-dir -r requirements-rerank.txt \
        && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
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