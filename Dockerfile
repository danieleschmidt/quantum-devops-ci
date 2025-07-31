# Multi-stage Dockerfile for quantum-devops-ci
# Base image with quantum computing frameworks and development tools

# Stage 1: Base quantum environment
FROM python:3.11-slim as quantum-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 quantum && \
    useradd --uid 1000 --gid quantum --shell /bin/bash --create-home quantum

# Stage 2: Development environment
FROM quantum-base as development

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy package files first for better caching
COPY package*.json ./
COPY pyproject.toml ./

# Install Node.js dependencies
RUN npm ci --only=production

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .[dev,all]

# Copy source code
COPY . .

# Change ownership to quantum user
RUN chown -R quantum:quantum /workspace

# Switch to non-root user
USER quantum

# Set up development environment
RUN pre-commit install

# Default command for development
CMD ["bash"]

# Stage 3: Production environment
FROM quantum-base as production

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY pyproject.toml ./

# Install production dependencies only
RUN npm ci --only=production && \
    pip install --upgrade pip setuptools wheel && \
    pip install -e .[qiskit,cirq,pennylane] --no-dev

# Copy source code
COPY src/ ./src/
COPY quantum-tests/ ./quantum-tests/
COPY examples/ ./examples/
COPY README.md LICENSE ./

# Change ownership
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import quantum_devops_ci; print('OK')" || exit 1

# Default command
CMD ["quantum-test", "--help"]

# Stage 4: CI/CD runner environment
FROM production as ci-runner

# Switch back to root for CI tools installation
USER root

# Install CI/CD tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install additional CI tools
RUN pip install --no-cache-dir \
    bandit[toml] \
    safety \
    pip-audit \
    coverage[toml]

# Install Node.js dev dependencies for CI
RUN npm install -g \
    eslint \
    prettier \
    @semantic-release/changelog \
    @semantic-release/git

# Copy CI configuration
COPY .pre-commit-config.yaml ./
COPY .eslintrc.json ./
COPY .prettierrc ./

# Switch back to quantum user
USER quantum

# Verify installations
RUN quantum-test --version && \
    quantum-lint --version && \
    python -c "import qiskit; print(f'Qiskit {qiskit.__version__}')" && \
    python -c "import cirq; print(f'Cirq {cirq.__version__}')" && \
    echo "All quantum frameworks installed successfully"

# CI/CD entry point
CMD ["quantum-test", "run", "--ci"]

# Multi-architecture build support
# docker buildx build --platform linux/amd64,linux/arm64 -t quantum-devops-ci .

# Build arguments for customization
ARG QUANTUM_FRAMEWORKS="qiskit,cirq,pennylane"
ARG PYTHON_VERSION="3.11"
ARG NODE_VERSION="18"

# Labels for metadata
LABEL maintainer="Quantum DevOps Community <community@quantum-devops.org>" \
      version="1.0.0" \
      description="Quantum DevOps CI/CD toolkit with support for Qiskit, Cirq, and PennyLane" \
      org.opencontainers.image.title="quantum-devops-ci" \
      org.opencontainers.image.description="CI/CD toolkit for quantum computing workflows" \
      org.opencontainers.image.url="https://quantum-devops-ci.readthedocs.io" \
      org.opencontainers.image.source="https://github.com/quantum-devops/quantum-devops-ci" \
      org.opencontainers.image.vendor="Quantum DevOps Community" \
      org.opencontainers.image.licenses="MIT"