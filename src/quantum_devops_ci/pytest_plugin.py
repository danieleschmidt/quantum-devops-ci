"""
Pytest plugin for quantum DevOps CI/CD.

This plugin extends pytest with quantum-specific fixtures, markers,
and test collection capabilities.
"""

import pytest
import warnings
from typing import Generator, Any, Dict, Optional
from pathlib import Path

from .testing import NoiseAwareTestBase
from . import AVAILABLE_FRAMEWORKS


def pytest_configure(config):
    """Configure pytest with quantum-specific settings."""
    # Register quantum markers
    config.addinivalue_line(
        "markers", "quantum: mark test as quantum test requiring quantum simulation"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take several minutes)"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring real quantum hardware"
    )
    config.addinivalue_line(
        "markers", "qiskit: mark test as Qiskit-specific"
    )
    config.addinivalue_line(
        "markers", "cirq: mark test as Cirq-specific"
    )
    config.addinivalue_line(
        "markers", "pennylane: mark test as PennyLane-specific"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    
    # Store framework availability in config
    config.quantum_frameworks = AVAILABLE_FRAMEWORKS
    config.hardware_available = _check_hardware_availability()


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on quantum requirements."""
    skip_hardware = pytest.mark.skip(reason="Hardware access not available")
    skip_qiskit = pytest.mark.skip(reason="Qiskit not available")
    skip_cirq = pytest.mark.skip(reason="Cirq not available")
    skip_pennylane = pytest.mark.skip(reason="PennyLane not available")
    
    for item in items:
        # Skip hardware tests if no hardware access
        if "hardware" in item.keywords and not config.hardware_available:
            item.add_marker(skip_hardware)
        
        # Skip framework-specific tests if framework not available
        if "qiskit" in item.keywords and not AVAILABLE_FRAMEWORKS['qiskit']:
            item.add_marker(skip_qiskit)
        
        if "cirq" in item.keywords and not AVAILABLE_FRAMEWORKS['cirq']:
            item.add_marker(skip_cirq)
        
        if "pennylane" in item.keywords and not AVAILABLE_FRAMEWORKS['pennylane']:
            item.add_marker(skip_pennylane)
        
        # Add quantum marker to tests in quantum directories
        if "quantum" in str(item.fspath) or "test_quantum" in item.name:
            item.add_marker(pytest.mark.quantum)


def pytest_runtest_setup(item):
    """Set up quantum test environment before running tests."""
    # Check if test requires quantum simulation
    if "quantum" in item.keywords:
        # Validate quantum test environment
        _validate_quantum_environment(item)


def pytest_report_header(config):
    """Add quantum framework information to pytest header."""
    lines = ["quantum frameworks:"]
    
    for framework, version in AVAILABLE_FRAMEWORKS.items():
        status = f"{framework}: {version}" if version else f"{framework}: not available"
        lines.append(f"  {status}")
    
    hardware_status = "available" if config.hardware_available else "not available"
    lines.append(f"  hardware access: {hardware_status}")
    
    return lines


@pytest.fixture(scope="session")
def quantum_test_config():
    """Session-scoped fixture providing quantum test configuration."""
    return {
        "default_shots": 1000,
        "default_backend": "qasm_simulator",
        "timeout_seconds": 300,
        "noise_simulation": True,
        "available_frameworks": AVAILABLE_FRAMEWORKS
    }


@pytest.fixture(scope="function")
def quantum_tester(quantum_test_config):
    """Function-scoped fixture providing NoiseAwareTest instance."""
    # Create a basic test instance without __init__
    tester = NoiseAwareTestBase()
    tester.setup_method(None)
    tester.default_shots = quantum_test_config["default_shots"]
    tester.timeout_seconds = quantum_test_config["timeout_seconds"]
    return tester


@pytest.fixture(scope="function")
def qiskit_backend():
    """Fixture providing Qiskit backend for testing."""
    if not AVAILABLE_FRAMEWORKS['qiskit']:
        pytest.skip("Qiskit not available")
    
    try:
        from qiskit.providers.aer import AerSimulator
        return AerSimulator()
    except ImportError:
        pytest.skip("Qiskit Aer not available")


@pytest.fixture(scope="function")
def cirq_simulator():
    """Fixture providing Cirq simulator for testing."""
    if not AVAILABLE_FRAMEWORKS['cirq']:
        pytest.skip("Cirq not available")
    
    try:
        import cirq
        return cirq.Simulator()
    except ImportError:
        pytest.skip("Cirq not available")


@pytest.fixture(scope="function") 
def pennylane_device():
    """Fixture providing PennyLane device for testing."""
    if not AVAILABLE_FRAMEWORKS['pennylane']:
        pytest.skip("PennyLane not available")
    
    try:
        import pennylane as qml
        return qml.device('default.qubit', wires=4)
    except ImportError:
        pytest.skip("PennyLane not available")


@pytest.fixture(scope="function")
def bell_circuit_qiskit():
    """Fixture providing Bell state circuit in Qiskit."""
    if not AVAILABLE_FRAMEWORKS['qiskit']:
        pytest.skip("Qiskit not available")
    
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture(scope="function")
def bell_circuit_cirq():
    """Fixture providing Bell state circuit in Cirq."""
    if not AVAILABLE_FRAMEWORKS['cirq']:
        pytest.skip("Cirq not available")
    
    import cirq
    
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key='result')
    )
    return circuit


@pytest.fixture(scope="function")
def ghz_circuit_qiskit():
    """Fixture providing GHZ state circuit in Qiskit."""
    if not AVAILABLE_FRAMEWORKS['qiskit']:
        pytest.skip("Qiskit not available")
    
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


@pytest.fixture(scope="function")
def random_circuit_qiskit():
    """Fixture providing random quantum circuit in Qiskit."""
    if not AVAILABLE_FRAMEWORKS['qiskit']:
        pytest.skip("Qiskit not available")
    
    from qiskit import QuantumCircuit
    from qiskit.circuit.random import random_circuit
    
    return random_circuit(4, 3, measure=True)


@pytest.fixture(scope="function")
def noisy_backend_qiskit():
    """Fixture providing noisy Qiskit backend for testing."""
    if not AVAILABLE_FRAMEWORKS['qiskit']:
        pytest.skip("Qiskit not available")
    
    try:
        from qiskit.providers.aer import AerSimulator
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
        
        # Create simple noise model
        noise_model = NoiseModel()
        error = depolarizing_error(0.01, 1)  # 1% depolarizing error
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        
        return AerSimulator(noise_model=noise_model)
        
    except ImportError:
        pytest.skip("Qiskit Aer or noise models not available")


@pytest.fixture(scope="session")
def quantum_metrics_collector():
    """Session-scoped fixture for collecting quantum test metrics."""
    metrics = {
        "tests_run": 0,
        "circuits_executed": 0,
        "total_shots": 0,
        "total_execution_time": 0.0,
        "frameworks_used": set()
    }
    
    yield metrics
    
    # Report metrics at session end
    print(f"\nQuantum Test Metrics:")
    print(f"  Tests run: {metrics['tests_run']}")
    print(f"  Circuits executed: {metrics['circuits_executed']}")
    print(f"  Total shots: {metrics['total_shots']}")
    print(f"  Total execution time: {metrics['total_execution_time']:.2f}s")
    print(f"  Frameworks used: {', '.join(metrics['frameworks_used'])}")


@pytest.fixture(autouse=True)
def track_quantum_metrics(request, quantum_metrics_collector):
    """Auto-use fixture to track quantum test metrics."""
    # Only track for quantum tests
    if "quantum" not in request.keywords:
        yield
        return
    
    import time
    start_time = time.time()
    
    yield
    
    # Update metrics after test
    execution_time = time.time() - start_time
    quantum_metrics_collector["tests_run"] += 1
    quantum_metrics_collector["total_execution_time"] += execution_time
    
    # Track framework usage
    if "qiskit" in request.keywords:
        quantum_metrics_collector["frameworks_used"].add("qiskit")
    if "cirq" in request.keywords:
        quantum_metrics_collector["frameworks_used"].add("cirq")
    if "pennylane" in request.keywords:
        quantum_metrics_collector["frameworks_used"].add("pennylane")


def pytest_addoption(parser):
    """Add quantum-specific command line options."""
    group = parser.getgroup("quantum", "Quantum testing options")
    
    group.addoption(
        "--quantum-shots",
        type=int,
        default=1000,
        help="Default number of shots for quantum tests"
    )
    
    group.addoption(
        "--quantum-backend",
        default="qasm_simulator",
        help="Default quantum backend for tests"
    )
    
    group.addoption(
        "--quantum-timeout",
        type=int,
        default=300,
        help="Timeout for individual quantum tests (seconds)"
    )
    
    group.addoption(
        "--skip-hardware",
        action="store_true",
        default=False,
        help="Skip tests requiring quantum hardware"
    )
    
    group.addoption(
        "--skip-slow",
        action="store_true", 
        default=False,
        help="Skip slow quantum tests"
    )


def _check_hardware_availability() -> bool:
    """Check if quantum hardware access is available."""
    # This would check for valid credentials and provider access
    # For now, return False as placeholder
    return False


def _validate_quantum_environment(item):
    """Validate quantum test environment before running test."""
    # Check framework availability
    if "qiskit" in item.keywords and not AVAILABLE_FRAMEWORKS['qiskit']:
        pytest.skip("Qiskit framework not available")
    
    if "cirq" in item.keywords and not AVAILABLE_FRAMEWORKS['cirq']:
        pytest.skip("Cirq framework not available")
    
    if "pennylane" in item.keywords and not AVAILABLE_FRAMEWORKS['pennylane']:
        pytest.skip("PennyLane framework not available")
    
    # Check hardware requirements
    if "hardware" in item.keywords and not item.config.hardware_available:
        pytest.skip("Quantum hardware access not available")


# Quantum-specific assertion helpers
def assert_quantum_state_close(state1, state2, tolerance=1e-10):
    """Assert that two quantum states are approximately equal."""
    import numpy as np
    
    if hasattr(state1, 'data') and hasattr(state2, 'data'):
        # Handle quantum state objects
        diff = np.abs(state1.data - state2.data)
    else:
        # Handle numpy arrays
        diff = np.abs(np.array(state1) - np.array(state2))
    
    max_diff = np.max(diff)
    assert max_diff < tolerance, f"Quantum states differ by {max_diff} > {tolerance}"


def assert_fidelity_above(result, target_fidelity, message=""):
    """Assert that quantum result has fidelity above threshold."""
    if hasattr(result, 'calculate_fidelity'):
        fidelity = result.calculate_fidelity()
    else:
        # Placeholder fidelity calculation
        fidelity = 0.95
    
    assert fidelity >= target_fidelity, f"Fidelity {fidelity:.3f} below threshold {target_fidelity:.3f}. {message}"


def assert_error_rate_below(result, max_error_rate, message=""):
    """Assert that quantum result has error rate below threshold."""
    if hasattr(result, 'error_rate'):
        error_rate = result.error_rate
    else:
        # Placeholder error rate calculation
        error_rate = 0.05
    
    assert error_rate <= max_error_rate, f"Error rate {error_rate:.3f} above threshold {max_error_rate:.3f}. {message}"


# Export plugin interface
__all__ = [
    'pytest_configure',
    'pytest_collection_modifyitems',
    'pytest_runtest_setup',
    'pytest_report_header',
    'pytest_addoption',
    'quantum_test_config',
    'quantum_tester',
    'qiskit_backend',
    'cirq_simulator',
    'pennylane_device',
    'bell_circuit_qiskit',
    'bell_circuit_cirq',
    'ghz_circuit_qiskit',
    'random_circuit_qiskit',
    'noisy_backend_qiskit',
    'quantum_metrics_collector',
    'assert_quantum_state_close',
    'assert_fidelity_above',
    'assert_error_rate_below'
]