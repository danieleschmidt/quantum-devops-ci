# User Guide

This guide helps you get started with the quantum-devops-ci toolkit and provides comprehensive information for using it in your quantum software development workflow.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Testing Quantum Algorithms](#testing-quantum-algorithms)
- [CI/CD Integration](#cicd-integration)
- [Cost Management](#cost-management)
- [Monitoring and Analytics](#monitoring-and-analytics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 5-Minute Setup

1. **Install the toolkit**:
   ```bash
   pip install quantum-devops-ci
   # or
   npm install -g @quantum-devops/cli
   ```

2. **Initialize your project**:
   ```bash
   quantum-devops init
   ```

3. **Configure your quantum providers**:
   ```bash
   quantum-devops config add-provider ibmq --token YOUR_TOKEN
   ```

4. **Run your first quantum test**:
   ```bash
   quantum-test run examples/
   ```

That's it! You now have quantum CI/CD running on your project.

## Installation

### System Requirements

- **Python**: 3.9 or higher
- **Node.js**: 18 or higher (for CLI features)
- **Operating Systems**: Linux, macOS, Windows (WSL recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 2GB for full installation with quantum simulators

### Installation Methods

#### Method 1: Python Package (Recommended)

```bash
# Install from PyPI
pip install quantum-devops-ci

# Install with all optional dependencies
pip install quantum-devops-ci[all]

# Install development version
pip install quantum-devops-ci[dev]
```

#### Method 2: Node.js CLI

```bash
# Install globally
npm install -g @quantum-devops/cli

# Verify installation
quantum-devops --version
```

#### Method 3: Docker Container

```bash
# Pull the official image
docker pull quantum-devops/ci:latest

# Run in current directory
docker run -v $(pwd):/workspace quantum-devops/ci:latest quantum-test run
```

#### Method 4: VS Code Dev Container

If you're using VS Code, you can use our pre-configured development container:

1. Install the Remote-Containers extension
2. Open your project in VS Code
3. Add `.devcontainer/devcontainer.json`:
   ```json
   {
     "image": "quantum-devops/devcontainer:latest",
     "features": {
       "quantum-frameworks": {"qiskit": "latest", "cirq": "latest"}
     }
   }
   ```
4. Rebuild and reopen in container

### Verification

Verify your installation:

```bash
# Check CLI version
quantum-devops --version

# Check Python package
python -c "import quantum_devops_ci; print(quantum_devops_ci.__version__)"

# Run system check
quantum-devops doctor
```

## Configuration

### Basic Configuration

The toolkit uses a hierarchical configuration system:

1. **Global config**: `~/.quantum-devops-ci/config.yml`
2. **Project config**: `quantum.config.yml`
3. **Environment variables**: Override any setting

#### Initial Setup

Run the configuration wizard:

```bash
quantum-devops config init
```

This will guide you through:
- Quantum provider setup
- Framework preferences
- Default testing parameters
- CI/CD integration options

#### Manual Configuration

Create `quantum.config.yml` in your project root:

```yaml
# quantum.config.yml
quantum_devops_ci:
  version: '1.0.0'
  framework: qiskit  # or cirq, pennylane
  
providers:
  ibmq:
    token: ${IBMQ_TOKEN}
    hub: ibm-q
    group: open
    project: main
  
  aws_braket:
    aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}
    region: us-east-1

testing:
  default_shots: 1000
  noise_simulation: true
  timeout_seconds: 300
  parallel_jobs: 4

cost_management:
  monthly_budget: 1000  # USD
  alerts:
    - threshold: 0.8  # 80% of budget
      action: warn
    - threshold: 0.95  # 95% of budget
      action: block

monitoring:
  metrics_enabled: true
  dashboard_url: https://your-dashboard.com
  retention_days: 90
```

### Provider Configuration

#### IBM Quantum

```bash
# Add IBM Quantum credentials
quantum-devops config add-provider ibmq \
  --token YOUR_IBMQ_TOKEN \
  --hub ibm-q \
  --group open \
  --project main
```

#### AWS Braket

```bash
# Add AWS Braket credentials
quantum-devops config add-provider aws-braket \
  --access-key YOUR_ACCESS_KEY \
  --secret-key YOUR_SECRET_KEY \
  --region us-east-1
```

#### Google Quantum AI

```bash
# Add Google Quantum AI credentials
quantum-devops config add-provider google-quantum \
  --project-id YOUR_PROJECT_ID \
  --credentials-file path/to/credentials.json
```

### Environment Variables

Override any configuration with environment variables:

```bash
export QUANTUM_DEVOPS_FRAMEWORK=qiskit
export QUANTUM_DEVOPS_DEFAULT_SHOTS=2000
export IBMQ_TOKEN=your_token_here
export AWS_ACCESS_KEY_ID=your_key_here
```

## Testing Quantum Algorithms

### Basic Testing

Create quantum tests using our testing framework:

```python
# tests/test_quantum_algorithm.py
import pytest
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture
from qiskit import QuantumCircuit

class TestBellState(NoiseAwareTest):
    
    @quantum_fixture
    def bell_circuit(self):
        """Create a Bell state circuit"""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
    
    def test_bell_state_ideal(self, bell_circuit):
        """Test Bell state without noise"""
        result = self.run_circuit(
            bell_circuit,
            shots=1000,
            backend='qasm_simulator'
        )
        
        # Verify we get roughly equal |00⟩ and |11⟩ states
        counts = result['counts']
        assert abs(counts.get('00', 0) - counts.get('11', 0)) < 100
        assert counts.get('01', 0) + counts.get('10', 0) < 50
    
    def test_bell_state_with_noise(self, bell_circuit):
        """Test Bell state under realistic noise"""
        results = self.run_with_noise_sweep(
            circuit=bell_circuit,
            noise_levels=[0.001, 0.01, 0.05],
            shots=2000
        )
        
        for noise_level, result in results.items():
            fidelity = self.calculate_bell_fidelity(result)
            expected_fidelity = 1.0 - 2 * noise_level  # Approximate
            assert fidelity > expected_fidelity * 0.8
    
    def test_hardware_compatibility(self, bell_circuit):
        """Test compatibility across different backends"""
        backends = ['qasm_simulator', 'statevector_simulator']
        
        for backend in backends:
            if self.backend_available(backend):
                result = self.run_circuit(bell_circuit, backend=backend)
                assert 'counts' in result or 'statevector' in result
```

### Advanced Testing Features

#### Parameterized Quantum Tests

```python
from quantum_devops_ci.testing import quantum_parametrize

@quantum_parametrize([
    {"n_qubits": 2, "depth": 5},
    {"n_qubits": 4, "depth": 10},
    {"n_qubits": 8, "depth": 15}
])
def test_algorithm_scaling(params):
    """Test algorithm performance with different parameters"""
    circuit = create_test_circuit(
        n_qubits=params["n_qubits"],
        depth=params["depth"]
    )
    
    result = run_circuit(circuit, shots=1000)
    
    # Verify results scale appropriately
    assert validate_scaling(result, params)
```

#### Error Mitigation Testing

```python
def test_error_mitigation(self, noisy_circuit):
    """Test effectiveness of error mitigation"""
    # Run without mitigation
    raw_result = self.run_noisy(noisy_circuit, noise_level=0.05)
    
    # Run with error mitigation
    mitigated_result = self.run_with_mitigation(
        noisy_circuit,
        noise_level=0.05,
        method="zero_noise_extrapolation"
    )
    
    raw_fidelity = self.calculate_fidelity(raw_result)
    mitigated_fidelity = self.calculate_fidelity(mitigated_result)
    
    # Mitigation should improve fidelity
    assert mitigated_fidelity > raw_fidelity * 1.1
```

### Running Tests

#### Command Line

```bash
# Run all quantum tests
quantum-test run

# Run specific test file
quantum-test run tests/test_quantum_algorithm.py

# Run with specific backend
quantum-test run --backend ibmq_qasm_simulator

# Run with custom shots
quantum-test run --shots 5000

# Run with noise simulation
quantum-test run --noise-model ibmq_16_melbourne

# Run tests in parallel
quantum-test run --parallel 4

# Generate coverage report
quantum-test run --coverage

# Run performance benchmarks
quantum-test benchmark
```

#### Programmatic Usage

```python
from quantum_devops_ci.testing import QuantumTestRunner

runner = QuantumTestRunner(
    framework='qiskit',
    backend='qasm_simulator',
    shots=1000
)

# Run single test
result = runner.run_test(test_function)

# Run test suite
results = runner.run_test_suite('tests/')

# Run with custom configuration
results = runner.run_with_config({
    'noise_simulation': True,
    'error_mitigation': 'readout_error',
    'optimization_level': 3
})
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/quantum-ci.yml`:

```yaml
name: Quantum CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quantum-test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Quantum Environment
        uses: quantum-devops/setup-action@v1
        with:
          python-version: '3.9'
          frameworks: 'qiskit,cirq'
          
      - name: Quantum Circuit Linting
        run: quantum-lint check src/
        
      - name: Run Quantum Tests
        env:
          IBMQ_TOKEN: ${{ secrets.IBMQ_TOKEN }}
        run: |
          quantum-test run \
            --framework qiskit \
            --backend qasm_simulator \
            --shots 1000 \
            --coverage
            
      - name: Performance Benchmarks
        run: quantum-test benchmark --compare-baseline
        
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: quantum-test-results
          path: test-results/
```

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
stages:
  - lint
  - test
  - benchmark
  - deploy

quantum-lint:
  stage: lint
  image: quantum-devops/ci:latest
  script:
    - quantum-lint check src/ --format junit
  artifacts:
    reports:
      junit: lint-results.xml

quantum-test:
  stage: test
  image: quantum-devops/ci:latest
  variables:
    QUANTUM_BACKEND: "qasm_simulator"
  script:
    - quantum-test run --parallel 4 --coverage
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - test-results/

quantum-benchmark:
  stage: benchmark
  image: quantum-devops/ci:latest
  script:
    - quantum-test benchmark --export-metrics
  artifacts:
    paths:
      - benchmarks/
```

### Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        IBMQ_TOKEN = credentials('ibmq-token')
        QUANTUM_BACKEND = 'qasm_simulator'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install quantum-devops-ci'
                sh 'quantum-devops config verify'
            }
        }
        
        stage('Lint') {
            steps {
                sh 'quantum-lint check src/ --format checkstyle > lint-results.xml'
                publishCheckstyle pattern: 'lint-results.xml'
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'quantum-test run tests/unit/ --junit-xml unit-results.xml'
                    }
                    post {
                        always {
                            junit 'unit-results.xml'
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        sh 'quantum-test run tests/integration/ --junit-xml integration-results.xml'
                    }
                    post {
                        always {
                            junit 'integration-results.xml'
                        }
                    }
                }
            }
        }
        
        stage('Benchmark') {
            when {
                branch 'main'
            }
            steps {
                sh 'quantum-test benchmark --store-results'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'benchmark-results',
                    reportFiles: 'index.html',
                    reportName: 'Quantum Benchmarks'
                ])
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test-results/**', fingerprint: true
        }
    }
}
```

## Cost Management

### Budget Configuration

Set up cost management in your configuration:

```yaml
# quantum.config.yml
cost_management:
  monthly_budget: 2000  # USD
  currency: USD
  
  # Cost alerts
  alerts:
    - threshold: 0.5   # 50% of budget
      action: notify
      recipients: [team@company.com]
    - threshold: 0.8   # 80% of budget
      action: warn
      recipients: [manager@company.com]
    - threshold: 0.95  # 95% of budget
      action: block
      
  # Spending rules
  rules:
    - name: development
      branches: [develop, feature/*]
      max_daily_spend: 50
      allowed_backends: [simulator]
      
    - name: staging
      branches: [staging]
      max_daily_spend: 200
      allowed_backends: [simulator, ibmq_qasm_simulator]
      
    - name: production
      branches: [main]
      max_daily_spend: 500
      allowed_backends: [all]
      requires_approval: true
```

### Cost Optimization

#### Automatic Optimization

```python
from quantum_devops_ci.cost import CostOptimizer

optimizer = CostOptimizer(
    monthly_budget=1000,
    optimization_goal="minimize_cost"  # or "minimize_time"
)

# Optimize a batch of experiments
experiments = [
    {"circuit": circuit1, "shots": 10000, "priority": "high"},
    {"circuit": circuit2, "shots": 5000, "priority": "medium"},
    {"circuit": circuit3, "shots": 1000, "priority": "low"}
]

optimized_plan = optimizer.optimize_experiments(
    experiments,
    constraints={
        "deadline": "2025-08-15T00:00:00",
        "max_queue_time": "2h"
    }
)

print(f"Estimated cost: ${optimized_plan.total_cost:.2f}")
print(f"Time savings: {optimized_plan.time_savings:.1f} hours")
```

#### Manual Cost Control

```bash
# Check current spending
quantum-cost status

# Get cost breakdown by provider
quantum-cost breakdown --by-provider

# Set spending limits
quantum-cost limit set --daily 100 --monthly 2000

# Get cost predictions
quantum-cost predict --experiments experiments.json

# Generate cost report
quantum-cost report --period last-month --format pdf
```

### Resource Scheduling

#### Intelligent Scheduling

```python
from quantum_devops_ci.scheduling import QuantumJobScheduler

scheduler = QuantumJobScheduler(
    optimization_goal="minimize_time",
    cost_constraint=500  # USD
)

jobs = [
    {"circuit": vqe_circuit, "shots": 8192, "deadline": "2025-08-03"},
    {"circuit": qaoa_circuit, "shots": 4096, "priority": "high"}
]

schedule = scheduler.create_optimal_schedule(
    jobs,
    available_backends=["ibmq_montreal", "ibmq_brooklyn", "aws_sv1"]
)

# Submit scheduled jobs
for job in schedule.jobs:
    job_id = scheduler.submit_job(job)
    print(f"Submitted job {job_id} to {job.backend}")
```

## Monitoring and Analytics

### Dashboard Setup

Configure monitoring dashboard:

```yaml
# quantum.config.yml
monitoring:
  enabled: true
  dashboard_url: "https://quantum-metrics.yourcompany.com"
  metrics:
    - execution_time
    - cost_per_job
    - success_rate
    - queue_time
    - fidelity_scores
  
  alerts:
    - metric: success_rate
      threshold: 0.95
      condition: below
      action: email
    - metric: cost_per_job
      threshold: 10.0
      condition: above
      action: slack
```

### Custom Metrics

Track custom metrics in your tests:

```python
from quantum_devops_ci.monitoring import MetricsCollector

collector = MetricsCollector()

def test_algorithm_performance():
    start_time = time.time()
    
    # Run your quantum algorithm
    result = run_quantum_algorithm()
    
    # Record metrics
    collector.record_metric('execution_time', time.time() - start_time)
    collector.record_metric('fidelity', calculate_fidelity(result))
    collector.record_metric('gate_count', count_gates(circuit))
    
    # Add custom tags
    collector.add_tags({
        'algorithm': 'VQE',
        'backend': 'ibmq_montreal',
        'noise_level': 0.01
    })
```

### Performance Analysis

Generate performance reports:

```bash
# Generate performance report
quantum-analytics report \
  --period last-30-days \
  --metrics execution_time,cost,fidelity \
  --format html

# Compare performance between branches
quantum-analytics compare \
  --baseline main \
  --compare feature/optimization \
  --metric execution_time

# Export metrics to external systems
quantum-analytics export \
  --format prometheus \
  --endpoint http://prometheus:9090
```

## Best Practices

### Test Organization

Structure your quantum tests for maintainability:

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_circuit_ops.py
│   └── test_optimization.py
├── integration/             # Cross-component tests
│   ├── test_full_pipeline.py
│   └── test_provider_integration.py
├── quantum/                 # Quantum-specific tests
│   ├── test_noise_models.py
│   ├── test_error_mitigation.py
│   └── test_hardware_compatibility.py
├── benchmarks/              # Performance tests
│   ├── test_scaling.py
│   └── test_optimization_comparison.py
└── fixtures/                # Test data
    ├── circuits/
    └── expected_results/
```

### Configuration Management

Use environment-specific configurations:

```
configs/
├── development.yml          # Local development
├── testing.yml             # CI/CD testing
├── staging.yml             # Staging environment
└── production.yml          # Production settings
```

### Version Control

Add appropriate `.gitignore` entries:

```gitignore
# Quantum DevOps CI
.quantum-devops-ci/
quantum-test-results/
*.qasm
*.log

# Credentials
.env
quantum-credentials.json

# Generated files
coverage.xml
benchmark-results/
```

### Security Best Practices

1. **Never commit credentials**:
   ```bash
   # Use environment variables or secret management
   export IBMQ_TOKEN="your-token"
   quantum-devops config add-provider ibmq --token $IBMQ_TOKEN
   ```

2. **Use least-privilege access**:
   ```yaml
   providers:
     ibmq:
       token: ${IBMQ_TOKEN}
       access_level: read-only  # For testing environments
   ```

3. **Enable audit logging**:
   ```yaml
   security:
     audit_logging: true
     log_level: INFO
     sensitive_data_masking: true
   ```

## Troubleshooting

### Common Issues

#### Provider Connection Errors

```bash
# Test provider connectivity
quantum-devops doctor --provider ibmq

# Reset provider configuration
quantum-devops config reset-provider ibmq

# Debug connection issues
QUANTUM_DEVOPS_DEBUG=true quantum-test run
```

#### Test Execution Failures

```python
# Enable detailed test logging
import logging
logging.getLogger('quantum_devops_ci').setLevel(logging.DEBUG)

# Use debug mode for test runner
runner = QuantumTestRunner(debug=True)
result = runner.run_test(my_test)
```

#### Performance Issues

```bash
# Profile test execution
quantum-test run --profile --export-profile profile.json

# Check resource usage
quantum-devops status --resources

# Optimize test configuration
quantum-test optimize-config tests/
```

### Getting Help

- **Documentation**: https://quantum-devops-ci.readthedocs.io
- **Community Forum**: https://community.quantum-devops.org
- **GitHub Issues**: https://github.com/quantum-devops/quantum-devops-ci/issues
- **Email Support**: support@quantum-devops.org (Enterprise customers)

### Debug Mode

Enable comprehensive debugging:

```bash
export QUANTUM_DEVOPS_DEBUG=true
export QUANTUM_DEVOPS_LOG_LEVEL=debug

quantum-test run --verbose
```

This will provide detailed information about:
- Configuration loading
- Provider connections
- Circuit compilation
- Job submission and monitoring
- Result processing
- Error details

---

This user guide should get you started with quantum-devops-ci. For more advanced topics and specific use cases, refer to our comprehensive documentation and example repositories.