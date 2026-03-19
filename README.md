# quantum-devops-ci

> A DevOps toolkit for quantum computing pipelines — noise-aware testing, CI/CD templates, and resource estimation. No external dependencies.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-91%20passing-brightgreen.svg)](tests/)
[![No Dependencies](https://img.shields.io/badge/dependencies-none%20(stdlib%20only)-success.svg)](pyproject.toml)

---

## What This Does

Quantum software has a DevOps problem. Circuits that look correct on paper can fail silently on real hardware because of noise, resource constraints, or subtle gate ordering bugs. This toolkit brings CI/CD discipline to quantum development:

| Problem | This Toolkit |
|---|---|
| Circuits degrade under noise | `NoisySimulator` + `NoiseAwareTestRunner` — test at multiple noise levels, flag failures |
| No budget checks before QPU submission | `ResourceEstimator` — qubit count, circuit depth, T-gate count, fault-tolerance overhead |
| No CI/CD templates for quantum | `CITemplate` — generates GitHub Actions workflows with noise-aware test steps |
| Circuit bugs found late | `QuantumCircuit.validate()` — catches gate-after-measure, empty circuits, etc. |

---

## Quick Start

```python
from quantum_devops import QuantumCircuit, NoiseAwareTestRunner, ResourceEstimator

# Build a Bell state circuit
qc = QuantumCircuit(2, name="Bell")
qc.h(0).cnot(0, 1).measure()

# Test at multiple noise levels
runner = NoiseAwareTestRunner(
    noise_levels=[0.0, 0.01, 0.05, 0.1],
    fidelity_threshold=0.85,
    shots=300,
)
report = runner.test_circuit(qc)
print(report.report())
# [PASS] Bell @ noise=0.000 fidelity=1.0000
# [PASS] Bell @ noise=0.010 fidelity=0.9988
# ...

# Check resource budget
est = ResourceEstimator()
resource_report, violations = est.budget_check(qc, max_depth=10, max_t_gates=50)
print(resource_report.summary())
```

---

## Installation

```bash
git clone https://github.com/danieleschmidt/quantum-devops-ci
cd quantum-devops-ci
pip install -e ".[dev]"  # stdlib only, no quantum framework deps needed
```

---

## Core Components

### `QuantumCircuit`

Simple, dependency-free circuit representation. Supports: `H`, `CNOT`, `RZ`, `RX`, `MEASURE`.

```python
from quantum_devops import QuantumCircuit
import math

qc = QuantumCircuit(4, name="QAOA")
qc.h(0).h(1).h(2).h(3)
qc.rz(0, math.pi / 4)      # RZ(π/4) = T gate
qc.cnot(0, 1).cnot(1, 2)
qc.rx(0, math.pi / 8)
qc.measure()

print(qc.draw())
# Circuit: QAOA (4 qubits)
# ──────────────────────────────────────────────────
#    0: H q[0]
#    1: H q[1]
#    ...
# Depth: 5  Gates: 9  2Q gates: 2

issues = qc.validate()  # [] = clean
```

### `NoisySimulator`

Depolarizing noise model using Pauli error injection. Computes fidelity vs the ideal (zero-noise) distribution.

```python
from quantum_devops import NoisySimulator

sim = NoisySimulator(error_rate=0.01, seed=42)
stats = sim.run_and_aggregate(qc, shots=500)
# {
#   "counts": {"00": 248, "11": 252},
#   "fidelity": 0.9988,
#   "error_rate": 0.01,
#   "shots": 500
# }
```

### `NoiseAwareTestRunner`

CI-style test harness. Runs circuits at multiple noise levels, checks fidelity, reports pass/fail per noise level.

```python
from quantum_devops import NoiseAwareTestRunner

runner = NoiseAwareTestRunner(
    noise_levels=[0.0, 0.001, 0.01, 0.05, 0.1],
    fidelity_threshold=0.90,
    shots=200,
    seed=42,
)

reports, all_passed = runner.test_suite([bell_circuit, ghz_circuit, qaoa_circuit])
runner.print_suite_summary(reports, all_passed)
```

Output includes per-circuit noise sensitivity classification:
- **ROBUST** — passes at noise ≥ 0.05
- **MODERATE** — passes at noise ≥ 0.01
- **SENSITIVE** — only passes at very low noise
- **CRITICAL** — fails at all tested noise levels

### `ResourceEstimator`

Analyzes circuits for hardware resource requirements. T-gate counting is critical for fault-tolerant quantum computing (FT-QC), where each T-gate requires expensive magic state distillation.

```python
from quantum_devops import ResourceEstimator

est = ResourceEstimator()
report = est.estimate(qc)
print(report.summary())
# Resource Report: QAOA
#   Qubits:          4
#   Circuit depth:   10
#   Total gates:     26
#   2Q gates:        6
#   T-gates:         8    ← RZ(π/4) gates, expensive in FT-QC
#   Clifford gates:  14   ← "free" gates in FT-QC
#   Est. runtime:    2.80 µs
#   FT overhead:     Low (few distillation rounds)

# Budget enforcement for CI
_, violations = est.budget_check(qc, max_depth=20, max_t_gates=50)
```

### `CITemplate`

Generates GitHub Actions YAML for quantum CI/CD pipelines.

```python
from quantum_devops import CITemplate

template = CITemplate(
    repo_name="my-quantum-project",
    python_versions=["3.11", "3.12"],
    noise_levels=[0.0, 0.01, 0.05, 0.1],
    fidelity_threshold=0.85,
    shots=300,
    enable_resource_estimation=True,
)

template.save(".github/workflows/quantum-ci.yml")
```

The generated workflow includes:
- Multi-Python matrix testing
- Noise-aware test execution at configured levels
- Resource estimation with T-gate budget enforcement
- PR comments with resource summary

---

## Running the Demo

```bash
python demo.py
```

Tests Bell state, GHZ-3, and QAOA-2L circuits at noise levels 0, 0.01, and 0.1, then generates a CI workflow.

---

## Running Tests

```bash
pytest tests/ -v
# 91 passed in ~0.2s
```

---

## Project Structure

```
quantum_devops/
  __init__.py       — public API
  circuit.py        — QuantumCircuit, Gate
  simulator.py      — NoisySimulator, QubitState
  testing.py        — NoiseAwareTestRunner, CircuitTestReport
  ci_template.py    — CITemplate (GitHub Actions YAML)
  resources.py      — ResourceEstimator, ResourceReport
examples/
  circuits.py       — Bell, GHZ, QAOA example circuits
tests/
  test_circuit.py
  test_simulator.py
  test_testing.py
  test_resources.py
  test_ci_template.py
demo.py             — end-to-end demonstration
```

---

## Why No External Dependencies?

Quantum frameworks (Qiskit, Cirq, PennyLane) have complex dependency trees and frequent breaking changes. This toolkit uses stdlib only — it works anywhere Python 3.10+ is installed, including CI runners, edge devices, and air-gapped environments.

The simulation model is intentionally simple: single-qubit statevectors with probabilistic Pauli error injection. For production quantum simulations, pair this toolkit with a proper simulator (Qiskit Aer, Cirq) and use these classes for CI/CD orchestration.

---

## License

MIT © Daniel Schmidt
