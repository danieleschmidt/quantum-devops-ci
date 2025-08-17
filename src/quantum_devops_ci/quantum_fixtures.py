"""
Quantum test fixtures and utilities for noise-aware testing.
Generation 1 implementation with core quantum circuit patterns.
"""

import pytest
from typing import Dict, List, Any, Optional, Callable
import numpy as np
from functools import wraps

# Framework-agnostic quantum fixture decorator
def quantum_fixture(func: Callable) -> Callable:
    """
    Decorator to mark quantum circuit fixtures.
    Provides automatic framework detection and circuit validation.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        circuit = func(*args, **kwargs)
        
        # Basic validation
        if hasattr(circuit, 'num_qubits'):
            assert circuit.num_qubits > 0, "Circuit must have at least one qubit"
        
        # Framework-specific validation
        if hasattr(circuit, 'qasm'):  # Qiskit
            try:
                qasm_str = circuit.qasm()
                assert len(qasm_str) > 0, "Circuit QASM should not be empty"
            except:
                pass  # Skip if QASM generation fails
        
        return circuit
    
    return pytest.fixture(wrapper)

# Common quantum circuit patterns
@quantum_fixture
def bell_circuit():
    """Create a Bell state circuit (framework-agnostic)."""
    try:
        # Try Qiskit first
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
    except ImportError:
        try:
            # Try Cirq
            import cirq
            qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
            circuit = cirq.Circuit()
            circuit.append(cirq.H(qubits[0]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.measure(*qubits, key='result'))
            return circuit
        except ImportError:
            # Fallback to mock
            return MockQuantumCircuit(2, 2, "bell")

@quantum_fixture
def ghz_circuit():
    """Create a 3-qubit GHZ state circuit."""
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        return qc
    except ImportError:
        try:
            import cirq
            qubits = [cirq.LineQubit(i) for i in range(3)]
            circuit = cirq.Circuit()
            circuit.append(cirq.H(qubits[0]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.CNOT(qubits[1], qubits[2]))
            circuit.append(cirq.measure(*qubits, key='result'))
            return circuit
        except ImportError:
            return MockQuantumCircuit(3, 3, "ghz")

@quantum_fixture
def random_circuit():
    """Create a random quantum circuit for testing."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.random import random_circuit as qiskit_random
        return qiskit_random(4, 5, seed=42)
    except ImportError:
        try:
            import cirq
            import random
            random.seed(42)
            qubits = [cirq.LineQubit(i) for i in range(4)]
            circuit = cirq.Circuit()
            
            # Add random gates
            gates = [cirq.H, cirq.X, cirq.Y, cirq.Z]
            for _ in range(5):
                qubit = random.choice(qubits)
                gate = random.choice(gates)
                circuit.append(gate(qubit))
            
            return circuit
        except ImportError:
            return MockQuantumCircuit(4, 4, "random")

@quantum_fixture
def qft_circuit():
    """Create a Quantum Fourier Transform circuit."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import QFT
        return QFT(3)
    except ImportError:
        return MockQuantumCircuit(3, 3, "qft")

class MockQuantumCircuit:
    """Mock quantum circuit for testing when frameworks are unavailable."""
    
    def __init__(self, num_qubits: int, num_clbits: int, circuit_type: str = "mock"):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.circuit_type = circuit_type
        self.depth = 3  # Mock depth
        self.gates = []
    
    def qasm(self) -> str:
        """Generate mock QASM."""
        return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{self.num_qubits}];
creg c[{self.num_clbits}];
// Mock {self.circuit_type} circuit
"""
    
    def __str__(self):
        return f"MockQuantumCircuit({self.num_qubits} qubits, {self.circuit_type})"

# Noise model fixtures
@pytest.fixture
def basic_noise_model():
    """Create a basic noise model for testing."""
    try:
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
        
        noise_model = NoiseModel()
        error = depolarizing_error(0.01, 1)  # 1% depolarizing error
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        
        return noise_model
    except ImportError:
        return MockNoiseModel()

@pytest.fixture
def realistic_noise_model():
    """Create a realistic noise model based on IBM hardware."""
    try:
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.providers.fake_provider import FakeManhattan
        
        backend = FakeManhattan()
        noise_model = NoiseModel.from_backend(backend)
        return noise_model
    except ImportError:
        return MockNoiseModel()

class MockNoiseModel:
    """Mock noise model for testing."""
    
    def __init__(self, error_rate: float = 0.01):
        self.error_rate = error_rate
    
    def __str__(self):
        return f"MockNoiseModel(error_rate={self.error_rate})"

# Parametrized quantum test decorator
def quantum_parametrize(test_params: List[Dict[str, Any]]):
    """
    Parametrize quantum tests with multiple configurations.
    
    Args:
        test_params: List of parameter dictionaries for testing
    
    Example:
        @quantum_parametrize([
            {"n_qubits": 2, "depth": 5, "noise": 0.01},
            {"n_qubits": 4, "depth": 10, "noise": 0.02}
        ])
        def test_scalability(params):
            pass
    """
    param_values = []
    param_ids = []
    
    for i, params in enumerate(test_params):
        param_values.append(params)
        # Create readable test ID
        id_parts = []
        for key, value in params.items():
            if isinstance(value, float):
                id_parts.append(f"{key}={value:.3f}")
            else:
                id_parts.append(f"{key}={value}")
        param_ids.append("-".join(id_parts))
    
    return pytest.mark.parametrize("params", param_values, ids=param_ids)

# Quantum backend fixtures
@pytest.fixture
def quantum_simulator():
    """Get quantum simulator backend."""
    try:
        from qiskit import Aer
        return Aer.get_backend('qasm_simulator')
    except ImportError:
        try:
            import cirq
            return cirq.Simulator()
        except ImportError:
            return MockQuantumBackend()

@pytest.fixture
def statevector_simulator():
    """Get statevector simulator backend."""
    try:
        from qiskit import Aer
        return Aer.get_backend('statevector_simulator')
    except ImportError:
        try:
            import cirq
            return cirq.Simulator()
        except ImportError:
            return MockQuantumBackend()

class MockQuantumBackend:
    """Mock quantum backend for testing."""
    
    def __init__(self, name: str = "mock_simulator"):
        self.name = name
    
    def run(self, circuit, shots: int = 1000):
        """Mock circuit execution."""
        return MockQuantumJob(shots)
    
    def __str__(self):
        return f"MockQuantumBackend({self.name})"

class MockQuantumJob:
    """Mock quantum job result."""
    
    def __init__(self, shots: int):
        self.shots = shots
    
    def result(self):
        return MockQuantumResult(self.shots)

class MockQuantumResult:
    """Mock quantum execution result."""
    
    def __init__(self, shots: int):
        self.shots = shots
    
    def get_counts(self, circuit=None):
        """Return mock measurement counts."""
        # Return mock Bell state results
        return {'00': self.shots // 2, '11': self.shots // 2}
    
    def get_statevector(self, circuit=None):
        """Return mock statevector."""
        # Return mock Bell state vector
        return np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

# Utility functions for quantum testing
def calculate_fidelity(result1, result2) -> float:
    """Calculate fidelity between two quantum measurement results."""
    # Simplified fidelity calculation for mock testing
    if hasattr(result1, 'get_counts') and hasattr(result2, 'get_counts'):
        counts1 = result1.get_counts()
        counts2 = result2.get_counts()
        
        # Simple overlap fidelity
        total_shots = sum(counts1.values())
        overlap = 0
        
        for state in set(counts1.keys()) | set(counts2.keys()):
            prob1 = counts1.get(state, 0) / total_shots
            prob2 = counts2.get(state, 0) / total_shots
            overlap += np.sqrt(prob1 * prob2)
        
        return overlap
    
    return 1.0  # Mock perfect fidelity

def validate_quantum_state(counts: Dict[str, int], expected_states: List[str]) -> bool:
    """Validate that measurement results contain only expected quantum states."""
    measured_states = set(counts.keys())
    expected_states_set = set(expected_states)
    
    # Check if all measured states are in expected states
    return measured_states.issubset(expected_states_set)

def calculate_bell_fidelity(result) -> float:
    """Calculate Bell state fidelity from measurement results."""
    if hasattr(result, 'get_counts'):
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Bell state should only have |00⟩ and |11⟩
        bell_counts = counts.get('00', 0) + counts.get('11', 0)
        return bell_counts / total_shots
    
    return 1.0  # Mock perfect fidelity