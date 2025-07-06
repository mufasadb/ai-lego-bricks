# Multi-stage Dockerfile for AI Lego Bricks

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install package in build mode
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ailego

# Set working directory
WORKDIR /app

# Copy installed package from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/ailego /usr/local/bin/ailego

# Copy source files needed at runtime
COPY ailego/ ./ailego/
COPY agent_orchestration/ ./agent_orchestration/
COPY chat/ ./chat/
COPY chunking/ ./chunking/
COPY llm/ ./llm/
COPY memory/ ./memory/
COPY pdf_to_text/ ./pdf_to_text/
COPY prompt/ ./prompt/
COPY tts/ ./tts/
COPY test/ ./test/

# Copy configuration files
COPY pyproject.toml requirements.txt ./

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/agents && \
    chown -R ailego:ailego /app

# Switch to non-root user
USER ailego

# Set environment variables
ENV PYTHONPATH=/app
ENV AILEGO_DATA_DIR=/app/data
ENV AILEGO_OUTPUT_DIR=/app/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ailego status || exit 1

# Default command
ENTRYPOINT ["ailego"]
CMD ["--help"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest black flake8 mypy

# Install additional debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to ailego user
USER ailego

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]