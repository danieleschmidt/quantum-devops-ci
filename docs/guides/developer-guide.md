# Developer Guide

This guide provides comprehensive information for developers contributing to or extending the quantum-devops-ci toolkit.

## Table of Contents

- [Getting Started](#getting-started)
- [Architecture Overview](#architecture-overview)
- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Testing Guidelines](#testing-guidelines)
- [Plugin Development](#plugin-development)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- Docker (for containerized development)
- Git

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/quantum-devops-ci.git
   cd quantum-devops-ci
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Set up Node.js environment**:
   ```bash
   npm install
   ```

4. **Run initial setup**:
   ```bash
   make setup-dev  # or: python setup.py develop
   ```

### VS Code Development Container

For the optimal development experience, use the provided dev container:

1. Install VS Code and the Remote-Containers extension
2. Open the repository in VS Code
3. When prompted, click "Reopen in Container"
4. The container will build with all necessary tools and dependencies

## Architecture Overview

### Core Components

```
quantum-devops-ci/
├── src/quantum_devops_ci/
│   ├── core/              # Core framework and abstractions
│   ├── testing/           # Testing framework and utilities
│   ├── scheduling/        # Job scheduling and resource management
│   ├── cost/              # Cost optimization and tracking
│   ├── monitoring/        # Metrics and performance monitoring
│   ├── linting/           # Code quality and circuit validation
│   ├── deployment/        # Deployment strategies and management
│   └── plugins/           # Plugin system and framework adapters
├── src/cli.js             # Node.js CLI interface
├── tests/                 # Test suites
├── docs/                  # Documentation
└── examples/              # Example implementations
```

### Plugin Architecture

The system uses a plugin-based architecture for extensibility:

```python
# Plugin interface example
class FrameworkPlugin:
    def supports(self, framework_name: str) -> bool:
        """Check if this plugin supports the given framework"""
        pass
    
    def create_circuit(self, circuit_data: dict) -> Any:
        """Create a framework-specific circuit object"""
        pass
    
    def execute_circuit(self, circuit: Any, backend: str, shots: int) -> dict:
        """Execute circuit and return results"""
        pass
```

## Development Setup

### Environment Configuration

Create a `.env` file in the project root:

```env
# Quantum provider credentials (for testing)
IBMQ_TOKEN=your_ibm_token_here
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
GOOGLE_QUANTUM_PROJECT_ID=your_google_project_id

# Development settings
DEBUG=true
LOG_LEVEL=debug
TEST_BACKEND=qasm_simulator
```

### Development Dependencies

The development environment includes:

- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Linting**: flake8, black, isort, mypy
- **Documentation**: sphinx, sphinx-rtd-theme
- **Quantum Frameworks**: qiskit, cirq, pennylane
- **Development Tools**: pre-commit, tox

### Pre-commit Hooks

Set up pre-commit hooks for code quality:

```bash
pre-commit install
```

This will run linting, formatting, and basic tests before each commit.

## Code Organization

### Module Structure

#### Core Module (`quantum_devops_ci.core`)

Contains base classes and interfaces:

```python
# quantum_devops_ci/core/base.py
from abc import ABC, abstractmethod

class QuantumBackend(ABC):
    @abstractmethod
    def execute(self, circuit, shots: int) -> dict:
        pass

class QuantumTest(ABC):
    @abstractmethod
    def setup_circuit(self) -> Any:
        pass
    
    @abstractmethod
    def validate_result(self, result: dict) -> bool:
        pass
```

#### Testing Module (`quantum_devops_ci.testing`)

Provides quantum-specific testing capabilities:

```python
# quantum_devops_ci/testing/noise_aware.py
class NoiseAwareTest:
    def run_with_noise_sweep(self, circuit, noise_levels, shots=1000):
        """Run circuit with multiple noise levels"""
        results = {}
        for level in noise_levels:
            noise_model = self.create_noise_model(level)
            result = self.execute_with_noise(circuit, noise_model, shots)
            results[level] = result
        return results
```

#### Plugin System (`quantum_devops_ci.plugins`)

Framework and provider adapters:

```python
# quantum_devops_ci/plugins/qiskit_adapter.py 
class QiskitAdapter(FrameworkPlugin):
    def supports(self, framework_name: str) -> bool:
        return framework_name.lower() == 'qiskit'
    
    def create_circuit(self, circuit_data: dict):
        # Convert generic circuit data to Qiskit QuantumCircuit
        pass
```

### Configuration Management

Configuration is managed through a hierarchical system:

1. **Default Configuration**: Built-in defaults
2. **User Configuration**: `~/.quantum-devops-ci/config.yml`
3. **Project Configuration**: `quantum.config.yml`
4. **Environment Variables**: Override any setting

```python
# quantum_devops_ci/config.py
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TestingConfig:
    default_shots: int = 1000
    noise_simulation: bool = True
    timeout_seconds: int = 300

@dataclass
class Config:
    testing: TestingConfig
    providers: Dict[str, dict]
    frameworks: List[str]
```

## Testing Guidelines

### Test Structure

Tests are organized by functionality:

```
tests/
├── unit/                  # Unit tests for individual components
├── integration/           # Integration tests across components
├── quantum/               # Quantum-specific test scenarios
├── fixtures/              # Test data and fixtures
└── conftest.py            # Pytest configuration
```

### Writing Quantum Tests

Use the testing framework for quantum-specific tests:

```python
# tests/quantum/test_noise_aware.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest

class TestQuantumAlgorithm(NoiseAwareTest):
    @pytest.fixture
    def bell_circuit(self):
        # Create Bell state circuit
        return self.create_bell_circuit()
    
    def test_bell_state_fidelity(self, bell_circuit):
        """Test Bell state fidelity under noise"""
        results = self.run_with_noise_sweep(
            bell_circuit, 
            noise_levels=[0.001, 0.01, 0.05]
        )
        
        for noise_level, result in results.items():
            fidelity = self.calculate_fidelity(result)
            assert fidelity > 0.8, f"Low fidelity at noise {noise_level}"
```

### Test Configuration

Use pytest fixtures for test configuration:

```python
# tests/conftest.py
import pytest
from quantum_devops_ci.config import Config

@pytest.fixture
def test_config():
    return Config(
        testing=TestingConfig(
            default_shots=100,  # Reduced for faster testing
            noise_simulation=False
        )
    )

@pytest.fixture
def mock_quantum_backend():
    """Mock backend for unit tests"""
    class MockBackend:
        def execute(self, circuit, shots):
            return {"counts": {"00": shots//2, "11": shots//2}}
    return MockBackend()
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quantum_devops_ci

# Run only unit tests
pytest tests/unit/

# Run specific test file
pytest tests/quantum/test_noise_aware.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Plugin Development

### Creating a Framework Plugin

To add support for a new quantum framework:

1. **Create the plugin class**:

```python
# quantum_devops_ci/plugins/my_framework.py
from quantum_devops_ci.core import FrameworkPlugin

class MyFrameworkPlugin(FrameworkPlugin):
    def supports(self, framework_name: str) -> bool:
        return framework_name.lower() == 'myframework'
    
    def create_circuit(self, circuit_data: dict):
        # Implementation specific to your framework
        pass
    
    def execute_circuit(self, circuit, backend: str, shots: int) -> dict:
        # Framework-specific execution logic
        pass
```

2. **Register the plugin**:

```python
# quantum_devops_ci/plugins/__init__.py
from .my_framework import MyFrameworkPlugin

FRAMEWORK_PLUGINS = [
    MyFrameworkPlugin(),
    # ... other plugins
]
```

3. **Add tests**:

```python
# tests/plugins/test_my_framework.py
import pytest
from quantum_devops_ci.plugins import MyFrameworkPlugin

class TestMyFrameworkPlugin:
    def test_supports_framework(self):
        plugin = MyFrameworkPlugin()
        assert plugin.supports('myframework')
        assert not plugin.supports('other')
```

### Creating a Provider Plugin

For adding new quantum cloud providers:

```python
# quantum_devops_ci/plugins/my_provider.py
from quantum_devops_ci.core import ProviderPlugin

class MyProviderPlugin(ProviderPlugin):
    def get_available_backends(self) -> List[str]:
        # Return list of available backends
        pass
    
    def submit_job(self, circuit, backend: str, shots: int) -> str:
        # Submit job and return job ID
        pass
    
    def get_job_result(self, job_id: str) -> dict:
        # Retrieve job results
        pass
```

## API Reference

### Core APIs

#### Testing Framework

```python
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture

class MyTest(NoiseAwareTest):
    def test_algorithm(self):
        # Your test implementation
        pass

@quantum_fixture
def my_circuit():
    # Circuit creation logic
    pass
```

#### Scheduling System

```python
from quantum_devops_ci.scheduling import QuantumJobScheduler

scheduler = QuantumJobScheduler(
    optimization_goal="minimize_cost"
)

schedule = scheduler.optimize_schedule(
    jobs=[...],
    constraints={...}
)
```

#### Cost Optimization

```python
from quantum_devops_ci.cost import CostOptimizer

optimizer = CostOptimizer(monthly_budget=1000)
plan = optimizer.optimize_experiments(experiments)
```

### CLI APIs

The CLI provides both command-line and programmatic interfaces:

```bash
# Command line usage
quantum-test run --framework qiskit --shots 1000
quantum-lint check src/
quantum-deploy submit --backend ibmq_qasm_simulator
```

```python
# Programmatic usage
from quantum_devops_ci.cli import QuantumCLI

cli = QuantumCLI()
result = cli.run_tests(framework='qiskit', shots=1000)
```

## Best Practices

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all public APIs
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

```python
from typing import Dict, List, Optional

def execute_quantum_circuit(
    circuit: Any,
    backend: str,
    shots: int = 1000,
    noise_model: Optional[Any] = None
) -> Dict[str, int]:
    """Execute a quantum circuit on the specified backend.
    
    Args:
        circuit: The quantum circuit to execute
        backend: Name of the quantum backend
        shots: Number of measurement shots
        noise_model: Optional noise model for simulation
        
    Returns:
        Dictionary containing measurement results
        
    Raises:
        QuantumExecutionError: If execution fails
    """
    pass
```

### Error Handling

Use custom exceptions for quantum-specific errors:

```python
# quantum_devops_ci/exceptions.py
class QuantumDevOpsError(Exception):
    """Base exception for quantum DevOps operations"""
    pass

class QuantumExecutionError(QuantumDevOpsError):
    """Raised when quantum circuit execution fails"""
    pass

class ResourceLimitError(QuantumDevOpsError):
    """Raised when resource limits are exceeded"""
    pass
```

### Logging

Use structured logging throughout the codebase:

```python
import logging
from quantum_devops_ci.logging import get_logger

logger = get_logger(__name__)

def execute_test(test_name: str):
    logger.info("Starting test execution", extra={
        "test_name": test_name,
        "operation": "test_execution"
    })
    
    try:
        # Test execution logic
        logger.info("Test completed successfully", extra={
            "test_name": test_name,
            "status": "success"
        })
    except Exception as e:
        logger.error("Test execution failed", extra={
            "test_name": test_name,
            "error": str(e),
            "status": "failed"
        })
        raise
```

### Performance Considerations

- Use async/await for I/O operations
- Implement caching for expensive operations
- Use connection pooling for quantum providers
- Profile performance-critical code paths

```python
import asyncio
from functools import lru_cache

class QuantumBackend:
    @lru_cache(maxsize=128)
    def get_backend_info(self, backend_name: str):
        """Cache backend information"""
        pass
    
    async def execute_async(self, circuit, shots: int):
        """Asynchronous circuit execution"""
        async with self.connection_pool.get_connection() as conn:
            return await conn.execute(circuit, shots)
```

## Debugging and Troubleshooting

### Debugging Tools

Enable debug mode for detailed logging:

```python
import os
os.environ['QUANTUM_DEVOPS_DEBUG'] = 'true'

from quantum_devops_ci import enable_debug_mode
enable_debug_mode()
```

### Common Issues

#### Circuit Execution Failures

```python
# Enable circuit debugging
from quantum_devops_ci.debugging import QuantumDebugger

debugger = QuantumDebugger()
result = debugger.run_with_inspection(
    circuit,
    inspect_state=True,
    save_intermediate_results=True
)
```

#### Provider Connection Issues

```python
# Test provider connectivity
from quantum_devops_ci.providers import test_connection

status = test_connection('ibmq')
if not status.connected:
    print(f"Connection failed: {status.error}")
```

#### Performance Issues

```python
# Profile quantum operations
from quantum_devops_ci.profiling import profile_quantum_operation

@profile_quantum_operation
def my_quantum_function():
    # Your quantum code here
    pass
```

### Logging Configuration

Configure logging for development:

```python
# logging.yml
version: 1
formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: detailed
    level: DEBUG
loggers:
  quantum_devops_ci:
    level: DEBUG
    handlers: [console]
```

### Testing Strategies

Use different testing strategies for different scenarios:

```python
# Unit tests with mocks
@pytest.fixture
def mock_quantum_provider():
    with patch('quantum_devops_ci.providers.IBMQProvider') as mock:
        yield mock

# Integration tests with real backends
@pytest.mark.integration
def test_real_backend():
    # Test with actual quantum backend
    pass

# Performance tests
@pytest.mark.performance
def test_large_circuit_performance():
    # Test performance with large circuits
    pass
```

---

This developer guide provides the foundation for contributing to and extending the quantum-devops-ci toolkit. For specific questions or advanced use cases, refer to the API documentation or reach out to the development team.