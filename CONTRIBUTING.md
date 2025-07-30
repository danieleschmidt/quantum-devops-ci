# Contributing to Quantum DevOps CI

Thank you for your interest in contributing to the Quantum DevOps CI project! This document provides guidelines and information for contributors who want to help improve quantum computing CI/CD practices.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Guidelines](#contribution-guidelines)
- [Quantum-Specific Guidelines](#quantum-specific-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Submission Process](#submission-process)
- [Community and Communication](#community-and-communication)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@quantum-devops-ci.org].

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.8+** with pip and virtual environment support
- **Node.js 16+** with npm
- **Git** for version control
- Access to quantum computing frameworks (Qiskit, Cirq, PennyLane)
- Basic understanding of quantum computing concepts
- Familiarity with CI/CD principles

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/quantum-devops/quantum-devops-ci.git
cd quantum-devops-ci

# Set up development environment
npm run setup

# Verify installation
quantum-devops-ci --version
quantum-test --version
```

## Development Environment

### Local Development Setup

1. **Install Dependencies**
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

2. **Configure Quantum Providers** (Optional)
```bash
# Set up environment variables for quantum hardware access
export IBMQ_TOKEN="your_token_here"
export AWS_ACCESS_KEY_ID="your_aws_key"
export GOOGLE_CLOUD_CREDENTIALS="path/to/credentials.json"

# Or create a local config file
cp config.example.yml config.local.yml
# Edit config.local.yml with your credentials
```

3. **Run Tests**
```bash
# Run all tests
npm test

# Run Python tests only
pytest

# Run quantum-specific tests (requires provider access)
pytest -m quantum
```

### Development Tools

We use several tools to maintain code quality:

- **Black** and **isort** for Python code formatting
- **ESLint** and **Prettier** for JavaScript code formatting
- **MyPy** for Python type checking
- **Pre-commit hooks** for automated checks
- **Pytest** for Python testing
- **Jest** for JavaScript testing

## Contribution Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement features, fix bugs, improve performance
4. **Documentation**: Improve guides, tutorials, API documentation
5. **Testing**: Add test cases, improve test coverage
6. **Examples**: Create usage examples and tutorials
7. **Templates**: Develop CI/CD templates for different scenarios

### Priority Areas

We particularly welcome contributions in these areas:

- **Framework Integration**: Support for additional quantum frameworks
- **Hardware Providers**: Integration with new quantum cloud services
- **Noise Models**: Enhanced noise simulation and characterization
- **Cost Optimization**: Algorithms for efficient resource allocation
- **IDE Integrations**: VS Code extensions, Jupyter plugins
- **Documentation**: Tutorials, best practices, case studies

## Quantum-Specific Guidelines

### Circuit Development Standards

When contributing quantum circuit code:

```python
# Good: Clear, documented quantum circuit
def create_bell_state(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Create a Bell state |00⟩ + |11⟩.
    
    Args:
        qc: Quantum circuit with at least 2 qubits
        
    Returns:
        Modified quantum circuit with Bell state preparation
    """
    qc.h(0)  # Create superposition
    qc.cx(0, 1)  # Entangle qubits
    return qc

# Bad: Unclear circuit without documentation
def bell(qc):
    qc.h(0)
    qc.cx(0, 1)
    return qc
```

### Testing Quantum Code

All quantum code must include appropriate tests:

```python
import pytest
from quantum_devops_ci.testing import NoiseAwareTest

class TestBellState(NoiseAwareTest):
    def test_bell_state_ideal(self):
        """Test Bell state in ideal conditions."""
        circuit = create_bell_state()
        result = self.run_circuit(circuit, shots=1000)
        
        # Check for correct state distribution
        counts = result.get_counts()
        assert '00' in counts
        assert '11' in counts
        assert abs(counts['00'] - counts['11']) < 100  # Rough equality
    
    @pytest.mark.slow
    def test_bell_state_noisy(self):
        """Test Bell state under realistic noise."""
        circuit = create_bell_state()
        result = self.run_with_noise(
            circuit, 
            noise_model='ibmq_essex',
            shots=8192
        )
        
        fidelity = self.calculate_state_fidelity(result, target_state='bell')
        assert fidelity > 0.8, f"Fidelity too low: {fidelity}"
```

### Hardware Resource Management

When writing code that uses quantum hardware:

- **Test on simulators first**: Always validate on simulators before hardware
- **Minimize hardware usage**: Use the minimum shots necessary for meaningful results
- **Respect rate limits**: Don't overwhelm quantum services with rapid requests
- **Handle errors gracefully**: Account for hardware failures and queue timeouts
- **Document costs**: Include estimated costs in documentation and examples

### Framework Compatibility

Ensure code works across quantum frameworks:

```python
# Good: Framework-agnostic approach
def optimize_circuit(circuit, backend_type="simulator"):
    """Optimize circuit for given backend type."""
    if isinstance(circuit, qiskit.QuantumCircuit):
        return optimize_qiskit_circuit(circuit, backend_type)
    elif isinstance(circuit, cirq.Circuit):
        return optimize_cirq_circuit(circuit, backend_type)
    else:
        raise NotImplementedError(f"Framework not supported: {type(circuit)}")

# Bad: Framework-specific code without abstraction
def optimize_circuit(qiskit_circuit):
    # Only works with Qiskit
    pass
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Quantum Tests**: Test quantum algorithms and circuits
4. **Hardware Tests**: Test with real quantum devices (optional)
5. **Performance Tests**: Benchmark and regression tests

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_utility_function():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Integration test."""
    pass

@pytest.mark.quantum
def test_quantum_algorithm():
    """Test requiring quantum simulation."""
    pass

@pytest.mark.slow
def test_comprehensive_benchmark():
    """Long-running test."""
    pass

@pytest.mark.hardware
@pytest.mark.skipif(not has_hardware_access(), reason="No hardware access")
def test_on_real_device():
    """Test requiring real quantum hardware."""
    pass
```

### Coverage Requirements

- **Minimum coverage**: 80% for all new code
- **Critical paths**: 100% coverage for core functionality
- **Quantum code**: Include both ideal and noisy simulations
- **Error handling**: Test failure modes and edge cases

## Documentation Standards

### Code Documentation

- **Docstrings**: Use Google-style docstrings for all public functions
- **Type hints**: Include type annotations for all parameters and returns
- **Examples**: Provide usage examples in docstrings
- **Complexity notes**: Document time/space complexity for algorithms

```python
def optimize_quantum_schedule(
    jobs: List[QuantumJob], 
    constraints: ResourceConstraints,
    optimization_goal: str = "minimize_cost"
) -> OptimizedSchedule:
    """
    Optimize quantum job scheduling for cost or time.
    
    Uses integer programming to find optimal job scheduling that respects
    hardware constraints and minimizes the specified objective function.
    
    Args:
        jobs: List of quantum jobs to schedule
        constraints: Resource constraints (budget, time, hardware)
        optimization_goal: Either "minimize_cost" or "minimize_time"
        
    Returns:
        Optimized schedule with job assignments and estimated metrics
        
    Raises:
        ValueError: If optimization_goal is not supported
        InfeasibleScheduleError: If no valid schedule exists
        
    Example:
        >>> jobs = [QuantumJob(circuit=my_circuit, shots=1000)]
        >>> constraints = ResourceConstraints(max_cost=100.0)
        >>> schedule = optimize_quantum_schedule(jobs, constraints)
        >>> print(f"Total cost: ${schedule.total_cost:.2f}")
        
    Note:
        This function has O(n²) complexity where n is the number of jobs.
        For large job sets (>1000), consider using the approximation algorithm.
    """
```

### README and Guides

- **Clear setup instructions**: Step-by-step installation and configuration
- **Working examples**: Complete, runnable code examples
- **Troubleshooting**: Common issues and solutions
- **Visual aids**: Diagrams, flowcharts, and screenshots where helpful

## Submission Process

### Before Submitting

1. **Run tests**: Ensure all tests pass locally
2. **Check formatting**: Run code formatters and linters
3. **Update documentation**: Include relevant documentation updates
4. **Test examples**: Verify that examples in documentation work
5. **Review changes**: Self-review your changes for clarity and correctness

### Pull Request Process

1. **Create a branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/noise-aware-testing
   git checkout -b fix/qiskit-version-compatibility
   git checkout -b docs/getting-started-tutorial
   ```

2. **Make focused commits**: Keep commits small and focused
   ```bash
   git commit -m "Add noise-aware test base class
   
   - Implement NoiseAwareTest with common noise models
   - Add helper methods for fidelity calculation
   - Include examples for Qiskit and Cirq frameworks"
   ```

3. **Update CHANGELOG**: Add entry describing your changes

4. **Submit PR**: Use the pull request template and provide:
   - Clear description of changes
   - Motivation and context
   - Testing performed
   - Screenshots (if applicable)
   - Breaking changes (if any)

### Review Process

All contributions go through code review:

1. **Automated checks**: CI/CD pipeline runs tests and checks
2. **Peer review**: At least one maintainer reviews the code
3. **Quantum expertise**: Quantum-specific changes reviewed by quantum experts
4. **Documentation review**: Documentation changes reviewed for clarity
5. **Integration testing**: Changes tested in full integration environment

### Merge Requirements

Before merging, ensure:

- [ ] All CI checks pass
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG updated
- [ ] No conflicts with main branch

## Community and Communication

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, technical discussions
- **GitHub Discussions**: General questions, brainstorming, community support
- **Discord/Slack**: Real-time chat (link provided in main README)
- **Email**: Direct contact with maintainers for sensitive issues

### Getting Help

- **Technical questions**: Use GitHub Discussions or Issues
- **Quantum computing concepts**: Reference quantum computing resources and textbooks
- **CI/CD best practices**: Consult DevOps documentation and industry standards
- **Framework-specific issues**: Check framework documentation (Qiskit, Cirq, etc.)

### Recognition

Contributors are recognized through:

- **CONTRIBUTORS.md**: All contributors listed in the repository
- **Release notes**: Major contributions highlighted in release announcements
- **Community showcase**: Outstanding contributions featured in community updates
- **Conference talks**: Contributors invited to present their work at events

## Development Workflow

### Branching Strategy

We use a modified Git Flow:

- **main**: Production-ready code
- **develop**: Integration branch for new features
- **feature/**: Feature development branches
- **fix/**: Bug fix branches
- **release/**: Release preparation branches
- **hotfix/**: Emergency fixes for production

### Release Process

1. **Feature freeze**: No new features added to develop
2. **Release candidate**: Create release branch and test thoroughly
3. **Documentation update**: Ensure all docs are current
4. **Version bump**: Update version numbers and changelog
5. **Tag release**: Create Git tag and GitHub release
6. **Deploy**: Update package registries (NPM, PyPI)
7. **Announce**: Notify community of new release

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project. See the [LICENSE](LICENSE) file for details.

## Questions?

If you have any questions about contributing, please:

1. Check this guide and existing documentation
2. Search existing GitHub Issues and Discussions
3. Create a new GitHub Discussion for general questions
4. Create a GitHub Issue for specific problems
5. Contact maintainers directly for sensitive matters

Thank you for contributing to the future of quantum DevOps! 🚀