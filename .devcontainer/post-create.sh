#!/bin/bash
# Post-create script for quantum-devops-ci development container

set -e

echo "🚀 Setting up quantum-devops-ci development environment..."

# Update package lists
sudo apt-get update

# Install additional system dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    jq \
    tree \
    graphviz \
    pandoc

# Set up Python environment
echo "🐍 Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev,all]"
else
    pip install -e .
fi

# Set up Node.js environment
echo "📦 Setting up Node.js environment..."
npm install

# Set up pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Create useful directories
mkdir -p logs temp quantum_results simulation_data benchmarks

# Set up aliases
cat >> ~/.bashrc << 'EOF'
# Quantum DevOps CI aliases
alias qd='quantum-devops'
alias qt='quantum-test'
alias ql='quantum-lint'
EOF

# Create welcome message
cat > ~/.welcome.txt << 'EOF'
⚛️  QUANTUM DEVOPS CI DEVELOPMENT ENVIRONMENT  ⚛️

Welcome to your quantum-powered development container!

🚀 Quick Start:
  • Run tests: npm test
  • Lint code: npm run lint
  • Run quantum tests: quantum-test run

Happy quantum coding! 🌟
EOF

echo "cat ~/.welcome.txt" >> ~/.bashrc

# Set proper permissions
chmod +x ~/.bashrc

echo "✅ Development environment setup complete!"