# quantum-devops-ci

> GitHub Actions templates to bring CI/CD discipline to Qiskit & Cirq workflows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45+-purple.svg)](https://qiskit.org/)
[![Cirq](https://img.shields.io/badge/Cirq-1.3+-blue.svg)](https://quantumai.google/cirq)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-Quantum-green.svg)](https://github.com/features/actions)

## 🌌 Overview

**quantum-devops-ci** addresses the critical DevOps gaps in quantum computing pipelines highlighted by IBM's February 2025 TechXchange post. This toolkit provides production-ready CI/CD templates, noise-aware testing frameworks, and hardware resource management for quantum software development.

## ✨ Key Features

- **Noise-Aware Unit Tests**: Automated testing under realistic quantum noise models
- **Pulse-Level Linting**: Catch hardware constraint violations before execution
- **Hardware Quota Scheduler**: Intelligent QPU time allocation and queuing
- **VS Code Dev Container**: Complete quantum development environment
- **Multi-Framework Support**: Works with Qiskit, Cirq, PennyLane, and more

## 🎯 Problem Solved

| Challenge | Traditional Approach | Our Solution |
|-----------|---------------------|--------------|
| QPU Access Costs | Manual scheduling | Automated quota management |
| Noise Validation | Post-execution discovery | Pre-execution simulation |
| Circuit Errors | Runtime failures | Static analysis & linting |
| Environment Setup | Hours of configuration | One-click dev container |

## 🚀 Quick Start

### Repository Setup

```bash
# Add quantum CI/CD to your project
npx quantum-devops-ci init

# This creates:
# - .github/workflows/quantum-ci.yml
# - .devcontainer/devcontainer.json
# - quantum-tests/
# - quantum.config.yml
```

### Basic GitHub Action

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
      
      - name: Quantum Circuit Linting
        uses: quantum-devops/lint-action@v1
        with:
          framework: qiskit
          checks:
            - circuit-depth
            - gate-compatibility
            - pulse-constraints
            - measurement-optimization
      
  quantum-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Quantum Environment
        uses: quantum-devops/setup-action@v1
        with:
          python-version: '3.9'
          frameworks: 'qiskit,cirq'
          simulators: 'aer,qsim'
      
      - name: Run Noise-Aware Tests
        run: |
          quantum-test run \
            --framework qiskit \
            --backend ibmq_qasm_simulator \
            --noise-model ibmq_manhattan \
            --shots 1000
      
      - name: Upload Test Results
        uses: actions/upload-artifact@v3
        with:
          name: quantum-test-results
          path: test-results/
```

## 🔧 Advanced Features

### Noise-Aware Testing

```python
# quantum_tests/test_quantum_algorithm.py
import pytest
from quantum_devops_ci import NoiseAwareTest, quantum_fixture
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel

@quantum_fixture
def bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

class TestQuantumAlgorithm(NoiseAwareTest):
    def test_bell_state_under_noise(self, bell_circuit):
        """Test Bell state preparation with realistic noise"""
        # Run with multiple noise levels
        results = self.run_with_noise_sweep(
            circuit=bell_circuit,
            noise_levels=[0.001, 0.01, 0.05],
            shots=8192
        )
        
        # Assert fidelity degradation is acceptable
        for noise_level, result in results.items():
            fidelity = self.calculate_bell_fidelity(result)
            assert fidelity > 0.8, f"Fidelity too low at noise level {noise_level}"
    
    def test_error_mitigation(self, bell_circuit):
        """Test error mitigation effectiveness"""
        # Compare with and without mitigation
        raw_result = self.run_noisy(bell_circuit, noise_level=0.05)
        mitigated_result = self.run_with_mitigation(
            bell_circuit,
            noise_level=0.05,
            method="zero_noise_extrapolation"
        )
        
        raw_fidelity = self.calculate_bell_fidelity(raw_result)
        mitigated_fidelity = self.calculate_bell_fidelity(mitigated_result)
        
        assert mitigated_fidelity > raw_fidelity * 1.1  # 10% improvement
```

### Pulse-Level Linting

```python
# .quantum-lint.yml
pulse_constraints:
  max_amplitude: 1.0
  min_pulse_duration: 16  # dt units
  phase_granularity: 0.01
  frequency_limits:
    - channel: d0
      min: -300e6
      max: 300e6

gate_constraints:
  allowed_gates:
    - name: cx
      qubits: [[0,1], [1,2], [2,3]]
    - name: rz
      qubits: all
  max_circuit_depth: 100
  max_two_qubit_gates: 50

# Lint custom pulse schedules
from quantum_devops_ci.linting import PulseLinter

linter = PulseLinter.from_config(".quantum-lint.yml")

# Analyze pulse schedule
issues = linter.lint_schedule(my_pulse_schedule)

for issue in issues:
    print(f"{issue.severity}: {issue.message} at t={issue.time}")
    if issue.suggestion:
        print(f"  Suggestion: {issue.suggestion}")
```

### Hardware Quota Management

```yaml
# quantum.config.yml
hardware_access:
  providers:
    - name: ibmq
      credentials_secret: IBMQ_TOKEN
      max_monthly_shots: 10_000_000
      priority_queue: research
      
    - name: aws_braket
      credentials_secret: AWS_CREDENTIALS
      devices:
        - name: Aria-1
          hourly_quota: 2
          max_circuit_depth: 20
          
quota_rules:
  - name: development
    branches: [develop, feature/*]
    max_shots_per_run: 1000
    allowed_backends: [simulator]
    
  - name: staging
    branches: [staging]
    max_shots_per_run: 10000
    allowed_backends: [simulator, ibmq_qasm_simulator]
    
  - name: production
    branches: [main]
    max_shots_per_run: 100000
    allowed_backends: [all]
    requires_approval: true
```

### Intelligent Scheduling

```python
from quantum_devops_ci.scheduling import QuantumJobScheduler

scheduler = QuantumJobScheduler(
    config_file="quantum.config.yml",
    optimization_goal="minimize_cost"  # or "minimize_time"
)

# Schedule batch of experiments
jobs = [
    {"circuit": vqe_circuit, "shots": 10000, "priority": "high"},
    {"circuit": qaoa_circuit, "shots": 5000, "priority": "medium"},
    {"circuit": test_circuit, "shots": 1000, "priority": "low"}
]

schedule = scheduler.optimize_schedule(
    jobs,
    constraints={
        "deadline": "2025-08-01T00:00:00",
        "budget": 1000.0,  # USD
        "preferred_devices": ["ibmq_manhattan", "ibmq_brooklyn"]
    }
)

print(f"Estimated cost: ${schedule.total_cost:.2f}")
print(f"Estimated time: {schedule.total_time_hours:.1f} hours")
print(f"Device allocation: {schedule.device_allocation}")
```

## 🐳 VS Code Dev Container

### Automatic Setup

```json
// .devcontainer/devcontainer.json
{
  "name": "Quantum Development Environment",
  "image": "quantum-devops/devcontainer:latest",
  "features": {
    "quantum-frameworks": {
      "qiskit": "0.45",
      "cirq": "1.3",
      "pennylane": "0.33"
    },
    "simulators": {
      "qiskit-aer": true,
      "cirq-qsim": true,
      "gpu-acceleration": true
    },
    "analysis-tools": {
      "jupyter": true,
      "quantum-visualization": true,
      "pulse-designer": true
    }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "quantum-devops.quantum-lint",
        "quantum-devops.circuit-visualizer",
        "quantum-devops.pulse-designer",
        "ms-python.python"
      ]
    }
  },
  "postCreateCommand": "quantum-devops setup-workspace"
}
```

### Integrated Development

```bash
# Open in dev container
code .

# Container includes:
# - Pre-configured quantum frameworks
# - Hardware emulators
# - Visualization tools
# - Debugging capabilities
# - Performance profilers
```

## 📊 CI/CD Dashboard

### Metrics Tracking

```python
from quantum_devops_ci.monitoring import QuantumCIMonitor

monitor = QuantumCIMonitor(
    project="quantum-algorithm-research",
    dashboard_url="https://quantum-ci-dashboard.company.com"
)

# Track build metrics
monitor.record_build({
    "commit": git_sha,
    "circuit_count": 45,
    "total_gates": 1823,
    "max_depth": 89,
    "estimated_fidelity": 0.923,
    "noise_tests_passed": 38,
    "noise_tests_total": 40
})

# Track hardware usage
monitor.record_hardware_usage({
    "backend": "ibmq_manhattan",
    "shots": 50000,
    "queue_time_minutes": 12.5,
    "execution_time_minutes": 3.2,
    "cost_usd": 15.75
})
```

### Automated Reporting

```yaml
# .github/workflows/quantum-report.yml
name: Weekly Quantum Metrics

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  generate-report:
    runs-on: ubuntu-latest
    steps:
      - name: Generate Quantum CI Report
        uses: quantum-devops/report-action@v1
        with:
          period: week
          include:
            - test-coverage
            - hardware-usage
            - cost-analysis
            - performance-trends
          
      - name: Send Report
        uses: quantum-devops/notify-action@v1
        with:
          channels: [email, slack]
          recipients: quantum-team@company.com
```

## 🧪 Testing Patterns

### Parameterized Quantum Tests

```python
import pytest
from quantum_devops_ci import quantum_parametrize

@quantum_parametrize([
    {"n_qubits": 2, "depth": 10, "noise": 0.01},
    {"n_qubits": 4, "depth": 20, "noise": 0.02},
    {"n_qubits": 8, "depth": 30, "noise": 0.03}
])
def test_scalability(params):
    """Test algorithm scalability with increasing system size"""
    circuit = create_test_circuit(
        n_qubits=params["n_qubits"],
        depth=params["depth"]
    )
    
    result = run_with_noise(
        circuit,
        noise_level=params["noise"],
        shots=1000
    )
    
    assert get_fidelity(result) > 0.8 - 0.05 * params["n_qubits"]
```

### Hardware Compatibility Tests

```python
from quantum_devops_ci import HardwareCompatibilityTest

class TestHardwareCompatibility(HardwareCompatibilityTest):
    def test_gate_decomposition(self, my_circuit):
        """Ensure circuit decomposes to native gates"""
        for backend in self.get_available_backends():
            decomposed = self.decompose_for_backend(my_circuit, backend)
            
            # Check all gates are native
            native_gates = self.get_native_gates(backend)
            for instruction in decomposed.data:
                assert instruction.operation.name in native_gates
            
            # Verify equivalence
            assert self.circuits_equivalent(my_circuit, decomposed)
```

## 🔌 Integration Examples

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - test
  - deploy

quantum-lint:
  stage: lint
  image: quantum-devops/ci:latest
  script:
    - quantum-lint check src/
    - quantum-lint check-pulses pulses/

quantum-test:
  stage: test
  image: quantum-devops/ci:latest
  script:
    - quantum-test run --parallel 4
    - quantum-test coverage --min 80

deploy-to-quantum:
  stage: deploy
  only:
    - main
  script:
    - quantum-deploy submit \
        --backend $QUANTUM_BACKEND \
        --experiments experiments/ \
        --wait-for-results
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Quantum Lint') {
            steps {
                sh 'quantum-lint check --format junit > lint-results.xml'
                junit 'lint-results.xml'
            }
        }
        
        stage('Quantum Tests') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'quantum-test unit --coverage'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'quantum-test integration --backend simulator'
                    }
                }
            }
        }
        
        stage('Hardware Validation') {
            when {
                branch 'main'
            }
            steps {
                sh 'quantum-test hardware --shots 1000 --timeout 3600'
            }
        }
    }
}
```

## 📈 Best Practices

### Circuit Optimization Pipeline

```python
from quantum_devops_ci.optimization import CircuitOptimizationPipeline

# Define optimization stages
pipeline = CircuitOptimizationPipeline([
    "remove_redundant_gates",
    "merge_rotations",
    "optimize_cx_chains",
    "layout_optimization",
    "pulse_optimization"
])

# Apply to circuit
optimized = pipeline.optimize(
    circuit=my_circuit,
    backend="ibmq_manhattan",
    optimization_level=3
)

# Compare metrics
metrics = pipeline.compare_circuits(my_circuit, optimized)
print(f"Gate reduction: {metrics.gate_reduction:.1%}")
print(f"Depth reduction: {metrics.depth_reduction:.1%}")
print(f"Expected fidelity improvement: {metrics.fidelity_gain:.3f}")
```

### Continuous Benchmarking

```yaml
# .github/workflows/quantum-benchmark.yml
name: Quantum Performance Benchmarks

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Quantum Benchmarks
        uses: quantum-devops/benchmark-action@v1
        with:
          suite: comprehensive
          backends:
            - qasm_simulator
            - statevector_simulator
          metrics:
            - execution_time
            - memory_usage
            - gate_count
            - circuit_depth
      
      - name: Compare with Baseline
        uses: quantum-devops/compare-action@v1
        with:
          baseline: main
          threshold: 5  # Alert if >5% regression
      
      - name: Update Dashboard
        if: github.ref == 'refs/heads/main'
        run: |
          quantum-benchmark push-results \
            --dashboard https://quantum-metrics.company.com \
            --project ${{ github.repository }}
```

## 🔍 Debugging Tools

### Quantum State Inspector

```python
from quantum_devops_ci.debugging import QuantumDebugger

debugger = QuantumDebugger()

# Set breakpoints in quantum circuit
debugger.set_breakpoint(circuit, after_gate=5)
debugger.set_breakpoint(circuit, after_gate="first_cx")

# Run with inspection
result = debugger.run(
    circuit,
    shots=1,
    inspect_state=True,
    inspect_entanglement=True
)

# Analyze state evolution
for breakpoint in result.breakpoints:
    print(f"\nAfter gate {breakpoint.gate_index}:")
    print(f"State vector: {breakpoint.statevector}")
    print(f"Entanglement: {breakpoint.entanglement_entropy:.3f}")
    print(f"Probability distribution: {breakpoint.probabilities}")
```

### Error Analysis

```python
from quantum_devops_ci.analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()

# Analyze error sources
error_report = analyzer.analyze_circuit(
    circuit=my_circuit,
    backend="ibmq_manhattan",
    error_sources=[
        "gate_errors",
        "readout_errors",
        "crosstalk",
        "decoherence"
    ]
)

# Generate mitigation strategy
mitigation = analyzer.suggest_mitigation(error_report)
print(f"Dominant error source: {error_report.dominant_source}")
print(f"Estimated error rate: {error_report.total_error_rate:.3%}")
print(f"Suggested mitigation: {mitigation.strategy}")
```

## 📊 Cost Optimization

### Budget-Aware Execution

```python
from quantum_devops_ci.cost import CostOptimizer

optimizer = CostOptimizer(
    monthly_budget=5000,  # USD
    priority_weights={
        "production": 0.5,
        "research": 0.3,
        "development": 0.2
    }
)

# Optimize experiment batch
experiments = load_experiment_queue()
optimized_plan = optimizer.optimize_experiments(
    experiments,
    constraints={
        "deadline": "2025-08-15",
        "min_shots": 1000,
        "max_queue_time": "2h"
    }
)

print(f"Total cost: ${optimized_plan.total_cost:.2f}")
print(f"Cost savings: ${optimized_plan.savings:.2f}")
print(f"Execution plan: {optimized_plan.schedule}")
```

## 🚀 Deployment Strategies

### Blue-Green Quantum Deployment

```yaml
# quantum-deployment.yml
deployment:
  strategy: blue_green
  environments:
    blue:
      backend: ibmq_manhattan
      allocation: 50%
    green:
      backend: ibmq_brooklyn
      allocation: 50%
  
  validation:
    min_fidelity: 0.95
    max_error_rate: 0.05
    comparison_shots: 10000
  
  rollout:
    canary_percentage: 10
    increment: 10
    wait_between: 1h
    rollback_on_failure: true
```

### A/B Testing Quantum Algorithms

```python
from quantum_devops_ci.deployment import QuantumABTest

ab_test = QuantumABTest(
    name="vqe_optimizer_comparison",
    variants={
        "A": {"optimizer": "COBYLA", "initial_point": "random"},
        "B": {"optimizer": "SPSA", "initial_point": "educated_guess"}
    },
    metrics=["convergence_rate", "final_energy", "total_evaluations"]
)

# Run A/B test
results = ab_test.run(
    circuit_factory=create_vqe_circuit,
    duration_hours=24,
    traffic_split=0.5
)

# Analyze results
winner = ab_test.determine_winner(
    results,
    confidence_level=0.95,
    minimum_difference=0.05
)

print(f"Winner: Variant {winner.variant}")
print(f"Improvement: {winner.improvement:.1%}")
print(f"Statistical significance: {winner.p_value:.4f}")
```

## 📚 Documentation

Full documentation: [https://quantum-devops-ci.readthedocs.io](https://quantum-devops-ci.readthedocs.io)

### Tutorials
- [Getting Started with Quantum CI/CD](docs/tutorials/01_getting_started.md)
- [Writing Noise-Aware Tests](docs/tutorials/02_noise_testing.md)
- [Managing QPU Resources](docs/tutorials/03_resource_management.md)
- [Quantum Debugging Techniques](docs/tutorials/04_debugging.md)

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional quantum framework support
- Enhanced noise models
- Cost optimization algorithms
- IDE integrations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@software{quantum_devops_ci,
  title={Quantum DevOps CI: Bringing Software Engineering to Quantum Computing},
  author={Daniel Schmidt},
  year={2025},
  url={https://github.com/danieleschmidt/quantum-devops-ci}
}
```

## 🏆 Acknowledgments

- IBM Quantum team for highlighting DevOps gaps
- Quantum software community
- GitHub Actions team

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
