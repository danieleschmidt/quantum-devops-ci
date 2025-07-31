#!/bin/bash

# Post-create script for quantum-devops-ci development container
# This script runs after the container is created to set up the development environment

set -e

echo "🚀 Setting up Quantum DevOps CI development environment..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt-get update

# Install additional development tools
echo "🔧 Installing additional development tools..."
sudo apt-get install -y --no-install-recommends \
    wget \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e .[dev,all]

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p \
    /workspace/test-results \
    /workspace/coverage-reports \
    /workspace/benchmark-results \
    /workspace/quantum-experiments \
    /workspace/.jupyter \
    /workspace/.vscode-server

# Set up Git configuration for development
echo "🔧 Setting up Git configuration..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main

# Download sample quantum circuits for testing
echo "🌌 Setting up quantum development resources..."
mkdir -p /workspace/sample-circuits
cat > /workspace/sample-circuits/bell_state.qasm << 'EOF'
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
EOF

cat > /workspace/sample-circuits/grover_2qubit.qasm << 'EOF'
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Initialize superposition
h q[0];
h q[1];

// Oracle for |11⟩
cz q[0],q[1];

// Diffuser
h q[0];
h q[1];
z q[0];
z q[1];
cz q[0],q[1];
h q[0];
h q[1];

measure q -> c;
EOF

# Set up Jupyter configuration
echo "📓 Configuring Jupyter Lab..."
mkdir -p /home/quantum/.jupyter
cat > /home/quantum/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.token = 'quantum-dev-token'
c.ServerApp.password = ''
c.ServerApp.allow_root = True
c.ServerApp.notebook_dir = '/workspace'
EOF

# Install Jupyter extensions for quantum computing
echo "🔌 Installing Jupyter extensions..."
pip install --quiet \
    jupyterlab-widgets \
    ipywidgets \
    qiskit[visualization] \
    matplotlib \
    seaborn \
    plotly

# Set up VSCode workspace settings
echo "⚙️ Setting up VSCode workspace..."
mkdir -p /workspace/.vscode
cat > /workspace/.vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Python: Quantum Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["quantum-tests/", "-v"],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Quantum CLI: Test",
            "type": "python",
            "request": "launch",
            "module": "quantum_devops_ci.cli",
            "args": ["test", "--help"],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }
    ]
}
EOF

# Create development aliases
echo "🔧 Setting up development aliases..."
cat >> /home/quantum/.bashrc << 'EOF'

# Quantum DevOps CI development aliases
alias qt='quantum-test'
alias ql='quantum-lint'
alias qb='quantum-benchmark'
alias qd='quantum-deploy'

# Common development commands
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'

# Python aliases
alias py='python'
alias pip3='pip'

# Testing aliases
alias test='npm test'
alias test-py='pytest'
alias test-quantum='quantum-test run'
alias coverage='coverage run -m pytest && coverage report'

# Linting aliases
alias lint='npm run lint'
alias lint-py='flake8 src/ tests/'
alias format='black src/ tests/ && prettier --write .'

# Development server aliases
alias serve-docs='cd docs && python -m http.server 8000'
alias jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

echo "🌌 Quantum DevOps CI development environment ready!"
echo "Available commands:"
echo "  qt (quantum-test) - Run quantum tests"
echo "  ql (quantum-lint) - Lint quantum circuits"
echo "  qb (quantum-benchmark) - Run benchmarks"
echo "  qd (quantum-deploy) - Deploy quantum experiments"
echo ""
echo "Development tools:"
echo "  npm test - Run all tests"
echo "  test-quantum - Run quantum-specific tests"
echo "  lint - Run all linters"
echo "  format - Format all code"
echo "  jupyter - Start Jupyter Lab"
echo ""
EOF

# Create a sample quantum development notebook
echo "📓 Creating sample development notebook..."
cat > /workspace/quantum-development.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Quantum DevOps CI Development Environment\n",
    "\n",
    "Welcome to your quantum development environment! This notebook demonstrates the available quantum computing frameworks and tools."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import quantum computing frameworks\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Qiskit\n",
    "from qiskit import QuantumCircuit, transpile, Aer, execute\n",
    "from qiskit.visualization import plot_histogram, plot_bloch_multivector\n",
    "\n",
    "# Quantum DevOps CI\n",
    "from quantum_devops_ci import quantum_fixture, NoiseAwareTest\n",
    "\n",
    "print(\"✅ All imports successful!\")\n",
    "print(\"🌌 Ready for quantum development!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a simple Bell state circuit\n",
    "qc = QuantumCircuit(2, 2)\n",
    "qc.h(0)\n",
    "qc.cx(0, 1)\n",
    "qc.measure_all()\n",
    "\n",
    "print(\"Bell State Circuit:\")\n",
    "print(qc.draw())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulate the circuit\n",
    "simulator = Aer.get_backend('qasm_simulator')\n",
    "job = execute(qc, simulator, shots=1000)\n",
    "result = job.result()\n",
    "counts = result.get_counts(qc)\n",
    "\n",
    "print(\"Measurement results:\")\n",
    "print(counts)\n",
    "\n",
    "# Visualize results\n",
    "plot_histogram(counts)\n",
    "plt.title(\"Bell State Measurement Results\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Verify installations
echo "🔍 Verifying installations..."

# Check Python packages
python -c "import quantum_devops_ci; print('✅ quantum_devops_ci installed')" || echo "❌ quantum_devops_ci not found"
python -c "import qiskit; print(f'✅ Qiskit {qiskit.__version__} installed')" || echo "❌ Qiskit not found"
python -c "import pytest; print('✅ pytest installed')" || echo "❌ pytest not found"

# Check Node.js packages
npm list --depth=0 2>/dev/null | head -5

# Check CLI tools
which quantum-test && echo "✅ quantum-test CLI available" || echo "❌ quantum-test CLI not found"
which pre-commit && echo "✅ pre-commit available" || echo "❌ pre-commit not found"

# Set proper permissions
echo "🔧 Setting permissions..."
sudo chown -R quantum:quantum /workspace
chmod +x /workspace/.devcontainer/post-create.sh

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Open a terminal and run 'qt --help' to see quantum testing options"
echo "2. Start Jupyter Lab with 'jupyter' command"
echo "3. Open quantum-development.ipynb to get started"
echo "4. Run 'npm test' to verify everything works"
echo ""
echo "Happy quantum coding! 🌌"
EOF