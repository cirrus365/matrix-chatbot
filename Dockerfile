FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Create directory for persistent configuration
RUN mkdir -p /app/data

# Environment variables (set at runtime)
ENV MATRIX_HOMESERVER=
ENV MATRIX_USERNAME=
ENV MATRIX_PASSWORD=
ENV OPENROUTER_API_KEY=
ENV JINA_API_KEY=

# Expose any necessary ports (if needed for webhooks/etc)
# EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Volume for persistent configuration
VOLUME ["/app/data"]

# Run the application
CMD ["python", "nifty.py"]
