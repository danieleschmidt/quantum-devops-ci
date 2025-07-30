# Development Guide

This guide explains how to set up a development environment for quantum-devops-ci and contribute to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Development Setup

### Prerequisites

- **Python 3.8+** with pip and venv
- **Node.js 16+** with npm
- **Git** for version control
- **Docker** (optional, for dev containers)

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/quantum-devops/quantum-devops-ci.git
cd quantum-devops-ci
```

2. **Set up Python environment**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,all]"
```

3. **Set up Node.js environment**:
```bash
# Install Node.js dependencies
npm install

# Install pre-commit hooks
npm run setup
```

4. **Configure development environment**:
```bash
# Copy example configuration
cp config.example.yml config.local.yml

# Set up environment variables
export QUANTUM_DEV_MODE=true
export QUANTUM_LOG_LEVEL=DEBUG
```

### IDE Setup

#### VS Code (Recommended)

The project includes VS Code configuration in `.vscode/`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

**Recommended Extensions**:
- Python
- Pylance
- Black Formatter
- GitLens
- Quantum Circuit Visualizer (if available)

#### Dev Container

Use the provided dev container for consistent development environment:

```bash
# With VS Code
code .
# Select "Reopen in Container" when prompted

# Or manually with Docker
docker build -t quantum-devops-dev .devcontainer/
docker run -it -v $(pwd):/workspace quantum-devops-dev
```

### Quantum Framework Setup

Install quantum frameworks for testing:

```bash
# Install all supported frameworks
pip install -e ".[qiskit,cirq,pennylane,braket]"

# Or install individually
pip install qiskit qiskit-aer
pip install cirq cirq-qsim  
pip install pennylane pennylane-lightning
pip install amazon-braket-sdk
```

### Provider Credentials (Optional)

For hardware testing, set up provider credentials:

```bash
# IBM Quantum
export IBMQ_TOKEN="your_token_here"

# AWS Braket  
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"

# Google Quantum AI
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

## Project Structure

```
quantum-devops-ci/
├── src/
│   ├── cli.js                    # Node.js CLI interface
│   └── quantum_devops_ci/        # Python package
│       ├── __init__.py
│       ├── testing.py            # Core testing framework
│       ├── linting.py            # Circuit linting
│       ├── scheduling.py         # Job scheduling
│       ├── monitoring.py         # Metrics collection
│       ├── cost.py               # Cost optimization
│       ├── deployment.py         # Deployment strategies
│       ├── cli.py                # Python CLI
│       └── pytest_plugin.py     # Pytest integration
├── examples/
│   ├── basic/                    # Basic usage examples
│   ├── advanced/                 # Advanced examples
│   └── frameworks/               # Framework-specific examples
├── quantum-tests/                # Test examples and utilities
│   ├── examples/
│   ├── integration/
│   └── unit/
├── tests/                        # Package tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                         # Documentation
│   ├── tutorials/
│   ├── api/
│   └── guides/
├── .github/                      # GitHub workflows
├── .devcontainer/                # Dev container config
├── package.json                  # Node.js package
├── pyproject.toml               # Python package
└── quantum.config.yml           # Example configuration
```

### Key Directories

- **`src/quantum_devops_ci/`**: Main Python package source code
- **`src/cli.js`**: Node.js CLI for project initialization
- **`tests/`**: Unit and integration tests for the package
- **`quantum-tests/`**: Example quantum tests and patterns
- **`examples/`**: Usage examples and demonstrations
- **`docs/`**: Documentation source files

## Development Workflow

### Branch Strategy

We use a modified Git Flow:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for new features
- **`feature/*`**: Feature development branches
- **`fix/*`**: Bug fix branches
- **`release/*`**: Release preparation branches

### Making Changes

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**:
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
```bash
# Run all tests
npm test

# Run specific test suites
pytest tests/unit/
pytest tests/integration/
quantum-test run examples/

# Run linting
npm run lint
```

4. **Commit your changes**:
```bash
git add .
git commit -m "feat: add noise-aware testing for Cirq

- Implement Cirq adapter for NoiseAwareTest
- Add noise model support for Cirq circuits
- Include integration tests with cirq-qsim
- Update documentation with Cirq examples"
```

5. **Push and create pull request**:
```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

**Examples**:
```
feat(testing): add support for PennyLane devices
fix(cost): resolve bulk discount calculation error
docs(tutorials): add getting started guide
test(integration): add hardware compatibility tests
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_testing.py     # Testing framework tests
│   ├── test_linting.py     # Linting engine tests
│   ├── test_cost.py        # Cost optimization tests
│   └── ...
├── integration/             # Integration tests
│   ├── test_qiskit.py      # Qiskit integration
│   ├── test_cirq.py        # Cirq integration
│   └── test_providers.py   # Provider integration
└── fixtures/                # Test fixtures
    ├── circuits/           # Sample quantum circuits
    ├── configs/            # Test configurations
    └── data/               # Test data files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quantum_devops_ci --cov-report=html

# Run specific test types
pytest -m "unit"              # Unit tests only
pytest -m "integration"       # Integration tests only  
pytest -m "not slow"          # Skip slow tests
pytest -m "qiskit"            # Qiskit-specific tests

# Run quantum example tests
quantum-test run quantum-tests/examples/

# Run with specific frameworks
pytest --framework=qiskit
pytest --framework=cirq
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_testing.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest

class TestNoiseAwareTest:
    def test_initialization(self):
        """Test NoiseAwareTest initialization."""
        tester = NoiseAwareTest(default_shots=2000)
        assert tester.default_shots == 2000
    
    def test_run_circuit_mock(self, mock_circuit):
        """Test circuit execution with mocked backend."""
        tester = NoiseAwareTest()
        result = tester.run_circuit(mock_circuit, shots=1000)
        assert result.shots == 1000
```

#### Integration Tests

```python
# tests/integration/test_qiskit.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest

@pytest.mark.qiskit
@pytest.mark.integration
class TestQiskitIntegration:
    def test_qiskit_circuit_execution(self):
        """Test actual Qiskit circuit execution."""
        qiskit = pytest.importorskip("qiskit")
        
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        tester = NoiseAwareTest()
        result = tester.run_circuit(qc, shots=1000)
        
        assert len(result.get_counts()) > 0
```

#### Quantum Algorithm Tests

```python
# quantum-tests/examples/test_algorithm.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture

@quantum_fixture
def vqe_circuit():
    """Create VQE test circuit."""
    # Implementation here
    pass

class TestVQEAlgorithm(NoiseAwareTest):
    @pytest.mark.quantum
    def test_vqe_convergence(self, vqe_circuit):
        """Test VQE algorithm convergence."""
        result = self.run_circuit(vqe_circuit, shots=4096)
        energy = self._calculate_energy(result)
        assert energy < -1.0  # Ground state energy
```

### Test Configuration

Configure pytest in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests", "quantum-tests"]
python_files = ["test_*.py", "*_test.py"]
markers = [
    "unit: unit tests",
    "integration: integration tests", 
    "slow: slow tests",
    "quantum: quantum algorithm tests",
    "qiskit: Qiskit-specific tests",
    "cirq: Cirq-specific tests",
    "pennylane: PennyLane-specific tests",
    "hardware: tests requiring quantum hardware"
]
```

## Code Quality

### Code Formatting

We use **Black** for Python code formatting:

```bash
# Format code
black src/ tests/ quantum-tests/

# Check formatting
black --check src/ tests/ quantum-tests/
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
```

### Import Sorting

We use **isort** for import organization:

```bash
# Sort imports
isort src/ tests/ quantum-tests/

# Check import sorting
isort --check-only src/ tests/ quantum-tests/
```

### Linting

We use **flake8** for linting:

```bash
# Run linting
flake8 src/ tests/ quantum-tests/
```

Configuration in `.flake8`:
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,.venv
```

### Type Checking

We use **mypy** for type checking:

```bash
# Run type checking
mypy src/quantum_devops_ci/
```

Configuration in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Pre-commit Hooks

Pre-commit hooks ensure code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
```

Install pre-commit hooks:
```bash
pre-commit install
```

## Documentation

### Documentation Structure

- **Tutorials**: Step-by-step guides for getting started
- **Guides**: How-to guides for specific tasks
- **API Reference**: Auto-generated API documentation
- **Architecture**: Design and architecture documentation

### Writing Documentation

#### Docstrings

Use Google-style docstrings:

```python
def run_circuit(
    self, 
    circuit: Any, 
    shots: Optional[int] = None,
    backend: str = "qasm_simulator"
) -> TestResult:
    """
    Run quantum circuit on simulator.
    
    Args:
        circuit: Quantum circuit to execute
        shots: Number of measurement shots
        backend: Backend name to use
        
    Returns:
        TestResult containing execution results
        
    Raises:
        ValueError: If circuit is invalid
        RuntimeError: If execution fails
        
    Example:
        >>> tester = NoiseAwareTest()
        >>> result = tester.run_circuit(my_circuit, shots=1000)
        >>> counts = result.get_counts()
    """
```

#### Markdown Documentation

Write clear, structured documentation:

```markdown
# Feature Name

Brief description of the feature.

## Usage

Show basic usage with code examples:

```python
from quantum_devops_ci import FeatureName

feature = FeatureName(config="value")
result = feature.execute()
```

## Advanced Usage

More complex examples and configuration options.

## API Reference

Link to auto-generated API docs.
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

## Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. **Update version numbers**:
   - `pyproject.toml`
   - `package.json`
   - `src/quantum_devops_ci/__init__.py`

2. **Update CHANGELOG.md**:
   - Add new version section
   - List all changes since last release
   - Follow [Keep a Changelog](https://keepachangelog.com/) format

3. **Run full test suite**:
```bash
npm test
pytest --cov=quantum_devops_ci
quantum-test run examples/
```

4. **Build and test packages**:
```bash
# Python package
python -m build
twine check dist/*

# Node.js package
npm pack
```

5. **Create release PR**:
   - Branch: `release/vX.Y.Z`
   - Include all version updates
   - Get approval from maintainers

6. **Tag and release**:
```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

7. **Publish packages**:
```bash
# Python to PyPI
twine upload dist/*

# Node.js to npm
npm publish
```

8. **Create GitHub release**:
   - Use tag vX.Y.Z
   - Include changelog content
   - Attach package files

### Automated Releases

GitHub Actions automates the release process:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build packages
        run: |
          python -m build
          npm pack
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
      
      - name: Publish to npm
        run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

## Troubleshooting

### Common Development Issues

**1. Import errors after installation**:
```bash
# Reinstall in development mode
pip uninstall quantum-devops-ci
pip install -e ".[dev,all]"
```

**2. Test failures with missing dependencies**:
```bash
# Install all test dependencies
pip install -e ".[dev,qiskit,cirq,pennylane]"
```

**3. Pre-commit hook failures**:
```bash
# Fix formatting issues
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

**4. Node.js CLI issues**:
```bash
# Reinstall Node.js dependencies
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/quantum-devops/quantum-devops-ci/issues)
- **Discussions**: [GitHub Discussions](https://github.com/quantum-devops/quantum-devops-ci/discussions)
- **Discord**: Join our developer community
- **Email**: dev@quantum-devops-ci.org

---

Thank you for contributing to quantum-devops-ci! Your contributions help bring software engineering discipline to quantum computing.