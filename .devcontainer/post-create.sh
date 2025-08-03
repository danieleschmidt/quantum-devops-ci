#!/bin/bash
set -e

echo "🚀 Setting up Quantum DevOps CI/CD Development Environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    sqlite3 \
    postgresql-client \
    redis-tools

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel

# Install project dependencies
echo "📚 Installing project dependencies..."
cd /workspace

# Install Python package in development mode
pip install -e .[dev,all]

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Initialize database
echo "🗄️ Initializing database..."
python3 -c "
from src.quantum_devops_ci.database.migrations import run_migrations

print('Running database migrations...')
success = run_migrations()
if success:
    print('✅ Database initialized successfully')
else:
    print('❌ Database initialization failed')
"

# Create quantum configuration
echo "⚙️ Creating quantum configuration..."
mkdir -p ~/.quantum_devops_ci
cat > ~/.quantum_devops_ci/config.yml << 'EOF'
# Quantum DevOps CI/CD Configuration
quantum_devops_ci:
  version: '1.0.0'
  environment: 'development'

# Database configuration
database:
  type: 'sqlite'
  path: '~/.quantum_devops_ci/quantum_devops.db'

# Cache configuration  
cache:
  type: 'memory'
  max_size: 1000

# Testing configuration
testing:
  default_backend: 'qasm_simulator'
  default_shots: 1000
  timeout_seconds: 300
  noise_simulation: true

# Development settings
development:
  debug: true
  auto_reload: true
  verbose_logging: true
EOF

# Create development aliases
echo "🔗 Creating development aliases..."
cat >> ~/.bashrc << 'EOF'

# Quantum DevOps CI/CD aliases
alias qtest='quantum-test run'
alias qlint='quantum-lint check'

echo "🚀 Quantum DevOps CI/CD Development Environment Ready!"
EOF

echo "✅ Development environment setup complete!"