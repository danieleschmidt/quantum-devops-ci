"""
Quantum tests package for quantum-devops-ci.

This package contains test examples and utilities for quantum DevOps CI/CD
testing patterns. It demonstrates best practices for testing quantum algorithms
with noise-aware simulation and hardware compatibility validation.
"""

# Test configuration
QUANTUM_TEST_CONFIG = {
    "default_shots": 1000,
    "default_backend": "qasm_simulator",
    "timeout_seconds": 300,
    "noise_simulation": True,
    "max_circuit_depth": 100,
    "supported_frameworks": ["qiskit", "cirq", "pennylane"]
}

# Export test utilities
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture
from quantum_devops_ci.pytest_plugin import (
    assert_quantum_state_close,
    assert_fidelity_above,
    assert_error_rate_below
)

__all__ = [
    "NoiseAwareTest",
    "quantum_fixture", 
    "assert_quantum_state_close",
    "assert_fidelity_above",
    "assert_error_rate_below",
    "QUANTUM_TEST_CONFIG"
]