# Getting Started with Quantum DevOps CI/CD

Welcome to the Quantum DevOps CI/CD toolkit! This tutorial will guide you through setting up quantum continuous integration and deployment for your quantum computing projects.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for version control
- Basic understanding of quantum computing concepts
- A quantum computing framework (Qiskit, Cirq, or PennyLane)

## Quick Setup

### 1. Initialize Your Project

Start by initializing quantum DevOps CI/CD in your project:

```bash
# Using NPX (recommended)
npx quantum-devops-ci init

# Or install globally first
npm install -g quantum-devops-ci
quantum-devops-ci init
```

This command will create:
- `quantum.config.yml` - Main configuration file
- `quantum-tests/` - Directory for quantum tests
- `.devcontainer/devcontainer.json` - VS Code dev container setup
- `.github/workflows/README.md` - CI/CD workflow documentation

### 2. Configure Your Quantum Provider

Edit `quantum.config.yml` to configure your quantum cloud provider:

```yaml
# quantum.config.yml
quantum_devops_ci:
  version: '1.0.0'
  framework: qiskit  # or cirq, pennylane
  provider: ibmq     # or aws-braket, google

hardware_access:
  providers:
    - name: ibmq
      credentials_secret: IBMQ_TOKEN
      max_monthly_shots: 10_000_000
      priority_queue: research

testing:
  default_shots: 1000
  noise_simulation: true
  timeout_minutes: 30

quota_rules:
  - name: development
    branches: [develop, feature/*]
    max_shots_per_run: 1000
    allowed_backends: [simulator]
  
  - name: production  
    branches: [main]
    max_shots_per_run: 100000
    allowed_backends: [all]
    requires_approval: true
```

### 3. Set Up Provider Credentials

Configure your quantum provider credentials as environment variables or GitHub secrets:

#### For IBM Quantum:
```bash
export IBMQ_TOKEN="your_ibm_quantum_token"
```

#### For AWS Braket:
```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

#### For Google Quantum AI:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### 4. Write Your First Quantum Test

Create a test file in the `quantum-tests/` directory:

```python
# quantum-tests/test_my_algorithm.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture
from qiskit import QuantumCircuit

@quantum_fixture
def bell_circuit():
    """Create a Bell state circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

class TestMyQuantumAlgorithm(NoiseAwareTest):
    @pytest.mark.quantum
    def test_bell_state_preparation(self, bell_circuit):
        """Test Bell state preparation."""
        # Run on ideal simulator
        result = self.run_circuit(bell_circuit, shots=1000)
        
        # Check results
        counts = result.get_counts()
        assert '00' in counts
        assert '11' in counts
        
        # Verify roughly equal distribution
        total = sum(counts.values())
        for state in ['00', '11']:
            if state in counts:
                prob = counts[state] / total
                assert 0.4 <= prob <= 0.6
    
    @pytest.mark.quantum
    @pytest.mark.slow
    def test_bell_state_with_noise(self, bell_circuit):
        """Test Bell state under noise."""
        result = self.run_with_noise(
            bell_circuit,
            noise_model='ibmq_essex',
            shots=4096
        )
        
        fidelity = self.calculate_bell_fidelity(result)
        assert fidelity > 0.8
```

### 5. Run Your Tests Locally

Test your quantum code locally before committing:

```bash
# Install Python dependencies
pip install -e ".[dev,qiskit]"

# Run quantum tests
quantum-test run

# Run with specific parameters
quantum-test run --framework qiskit --backend qasm_simulator --shots 2000

# Run with noise simulation
quantum-test run --noise-model ibmq_essex

# Lint quantum circuits
quantum-test lint

# Check cost estimates
quantum-test cost --budget 1000 --experiments experiments.json
```

### 6. Set Up GitHub Actions

The initialization created workflow documentation in `.github/workflows/README.md`. Based on this, create your CI/CD workflow:

```yaml
# .github/workflows/quantum-ci.yml
name: Quantum CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quantum-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,qiskit]"
      
      - name: Quantum Circuit Linting
        run: |
          quantum-test lint --check circuits --check pulses
      
  quantum-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,qiskit]"
      
      - name: Run Quantum Tests
        env:
          IBMQ_TOKEN: ${{ secrets.IBMQ_TOKEN }}
        run: |
          quantum-test run --framework qiskit --coverage
      
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
      
  cost-analysis:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v3
      
      - name: Cost Impact Analysis
        run: |
          quantum-test cost --budget 5000 --forecast
```

### 7. Advanced Configuration

#### Custom Noise Models

Configure custom noise models in your test configuration:

```python
# quantum-tests/conftest.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest

@pytest.fixture
def custom_noise_tester():
    """Custom noise-aware tester with specific configuration."""
    tester = NoiseAwareTest(
        default_shots=2000,
        timeout_seconds=600
    )
    
    # Configure custom noise models
    tester.add_noise_model('custom_low', depolarizing_error=0.001)
    tester.add_noise_model('custom_high', depolarizing_error=0.05)
    
    return tester
```

#### Hardware Scheduling

Configure hardware job scheduling:

```yaml
# quantum.config.yml
scheduling:
  optimization_goal: minimize_cost  # or minimize_time
  
  constraints:
    max_queue_time_hours: 4
    preferred_times:
      - start: "02:00"
        end: "06:00"
        timezone: "UTC"
    
  fallback_strategy:
    use_simulator: true
    max_fallback_shots: 10000
```

#### Cost Optimization

Set up automated cost optimization:

```yaml
# quantum.config.yml
cost_optimization:
  monthly_budget: 5000  # USD
  
  rules:
    - condition: "branch == 'develop'"
      max_cost_per_job: 10
      auto_approve: true
    
    - condition: "branch == 'main'"
      max_cost_per_job: 500
      requires_approval: true
      
  alerts:
    budget_threshold: 0.8  # Alert at 80% budget usage
    email: "quantum-team@company.com"
```

## Next Steps

Now that you have the basics set up:

1. **[Write Noise-Aware Tests](02_noise_testing.md)** - Learn advanced testing patterns
2. **[Manage QPU Resources](03_resource_management.md)** - Optimize hardware usage
3. **[Debug Quantum Circuits](04_debugging.md)** - Use quantum debugging tools
4. **[Set Up Monitoring](05_monitoring.md)** - Track metrics and performance

## Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
# Install in development mode
pip install -e ".[dev,all]"

# Or install specific framework
pip install -e ".[qiskit]"
```

**2. Quantum provider authentication**
```bash
# Verify credentials
quantum-test validate --provider ibmq

# Test connection
quantum-test status
```

**3. Test timeouts**
```yaml
# Increase timeout in quantum.config.yml
testing:
  timeout_minutes: 60  # Increase from default 30
```

**4. Hardware quota exceeded**
Check your usage:
```bash
quantum-test monitor --project my-project --summary
```

### Getting Help

- **Documentation**: [https://quantum-devops-ci.readthedocs.io](https://quantum-devops-ci.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/quantum-devops/quantum-devops-ci/issues)
- **Discussions**: [GitHub Discussions](https://github.com/quantum-devops/quantum-devops-ci/discussions)
- **Community**: Join our Discord server for real-time help

## Example Projects

Check out these example projects to see quantum DevOps CI/CD in action:

- [VQE Molecular Simulation](https://github.com/quantum-devops/vqe-example)
- [QAOA Optimization](https://github.com/quantum-devops/qaoa-example)  
- [Quantum Machine Learning](https://github.com/quantum-devops/qml-example)
- [Multi-Framework Project](https://github.com/quantum-devops/multi-framework-example)

---

**Congratulations!** You've successfully set up quantum DevOps CI/CD for your project. Your quantum algorithms now have the same software engineering discipline as classical software, with automated testing, cost optimization, and deployment capabilities.