"""
Fixtures for common quantum circuits used in testing.

Provides pre-defined quantum circuits for various algorithms and test scenarios.
"""

import pytest
from typing import Dict, List, Any
import numpy as np


class QuantumCircuitFixtures:
    """Collection of quantum circuit fixtures for testing."""
    
    @staticmethod
    def bell_state_circuit() -> Dict[str, Any]:
        """Create a Bell state circuit fixture."""
        return {
            'name': 'bell_state',
            'description': 'Creates a maximally entangled Bell state |00⟩ + |11⟩',
            'qubits': 2,
            'classical_bits': 2,
            'operations': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cx', 'qubits': [0, 1]},
                {'gate': 'measure', 'qubits': [0, 1], 'classical': [0, 1]}
            ],
            'expected_results': {
                'ideal': {'00': 0.5, '11': 0.5},
                'noise_0.01': {'00': 0.48, '01': 0.02, '10': 0.02, '11': 0.48},
                'noise_0.05': {'00': 0.42, '01': 0.08, '10': 0.08, '11': 0.42}
            }
        }
    
    @staticmethod
    def ghz_state_circuit(num_qubits: int = 3) -> Dict[str, Any]:
        """Create a GHZ state circuit fixture."""
        operations = [{'gate': 'h', 'qubits': [0]}]
        
        # Add controlled-X gates
        for i in range(1, num_qubits):
            operations.append({'gate': 'cx', 'qubits': [0, i]})
        
        # Add measurements
        operations.append({
            'gate': 'measure',
            'qubits': list(range(num_qubits)),
            'classical': list(range(num_qubits))
        })
        
        # Expected results for 3-qubit GHZ
        expected_3qubit = {
            'ideal': {'000': 0.5, '111': 0.5},
            'noise_0.01': {'000': 0.47, '001': 0.01, '010': 0.01, '100': 0.01, '111': 0.47, '110': 0.01, '101': 0.01, '011': 0.01},
        }
        
        return {
            'name': f'ghz_state_{num_qubits}q',
            'description': f'Creates a {num_qubits}-qubit GHZ state',
            'qubits': num_qubits,
            'classical_bits': num_qubits,
            'operations': operations,
            'expected_results': expected_3qubit if num_qubits == 3 else {}
        }
    
    @staticmethod
    def quantum_fourier_transform_circuit(num_qubits: int = 3) -> Dict[str, Any]:
        """Create a Quantum Fourier Transform circuit fixture."""
        operations = []
        
        # QFT implementation
        for j in range(num_qubits):
            operations.append({'gate': 'h', 'qubits': [j]})
            
            for k in range(j + 1, num_qubits):
                # Controlled rotation
                angle = np.pi / (2 ** (k - j))
                operations.append({
                    'gate': 'cp',
                    'qubits': [k, j],
                    'parameters': [angle]
                })
        
        # Swap qubits to reverse the order
        for i in range(num_qubits // 2):
            operations.append({
                'gate': 'swap',
                'qubits': [i, num_qubits - 1 - i]
            })
        
        operations.append({
            'gate': 'measure_all'
        })
        
        return {
            'name': f'qft_{num_qubits}q',
            'description': f'Quantum Fourier Transform on {num_qubits} qubits',
            'qubits': num_qubits,
            'classical_bits': num_qubits,
            'operations': operations,
            'expected_results': {
                'ideal': {format(i, f'0{num_qubits}b'): 1/2**num_qubits for i in range(2**num_qubits)}
            }
        }
    
    @staticmethod
    def vqe_ansatz_circuit(num_qubits: int = 4, depth: int = 2) -> Dict[str, Any]:
        """Create a VQE ansatz circuit fixture."""
        operations = []
        parameter_count = 0
        
        for layer in range(depth):
            # Single qubit rotations
            for qubit in range(num_qubits):
                operations.extend([
                    {'gate': 'ry', 'qubits': [qubit], 'parameters': [f'theta_{parameter_count}']},
                    {'gate': 'rz', 'qubits': [qubit], 'parameters': [f'theta_{parameter_count + 1}']}
                ])
                parameter_count += 2
            
            # Entangling gates
            for qubit in range(num_qubits - 1):
                operations.append({'gate': 'cx', 'qubits': [qubit, qubit + 1]})
        
        operations.append({'gate': 'measure_all'})
        
        return {
            'name': f'vqe_ansatz_{num_qubits}q_d{depth}',
            'description': f'VQE ansatz with {num_qubits} qubits and depth {depth}',
            'qubits': num_qubits,
            'classical_bits': num_qubits,
            'operations': operations,
            'parameters': [f'theta_{i}' for i in range(parameter_count)],
            'parameter_bounds': [(-np.pi, np.pi) for _ in range(parameter_count)]
        }
    
    @staticmethod
    def qaoa_circuit(num_qubits: int = 4, p: int = 1) -> Dict[str, Any]:
        """Create a QAOA circuit fixture."""
        operations = []
        
        # Initial state preparation (superposition)
        for qubit in range(num_qubits):
            operations.append({'gate': 'h', 'qubits': [qubit]})
        
        # QAOA layers
        for layer in range(p):
            # Problem Hamiltonian (example: Max-Cut)
            for qubit in range(num_qubits - 1):
                operations.append({
                    'gate': 'rzz',
                    'qubits': [qubit, qubit + 1],
                    'parameters': [f'gamma_{layer}']
                })
            
            # Mixer Hamiltonian
            for qubit in range(num_qubits):
                operations.append({
                    'gate': 'rx',
                    'qubits': [qubit],
                    'parameters': [f'beta_{layer}']
                })
        
        operations.append({'gate': 'measure_all'})
        
        return {
            'name': f'qaoa_{num_qubits}q_p{p}',
            'description': f'QAOA circuit with {num_qubits} qubits and {p} layers',
            'qubits': num_qubits,
            'classical_bits': num_qubits,
            'operations': operations,
            'parameters': [f'gamma_{i}' for i in range(p)] + [f'beta_{i}' for i in range(p)]
        }
    
    @staticmethod
    def random_circuit(num_qubits: int = 3, depth: int = 10, gate_set: List[str] = None) -> Dict[str, Any]:
        """Create a random quantum circuit fixture."""
        if gate_set is None:
            gate_set = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']
        
        operations = []
        np.random.seed(42)  # For reproducible random circuits
        
        for _ in range(depth):
            gate = np.random.choice(gate_set)
            
            if gate in ['h', 'x', 'y', 'z']:
                # Single qubit gate
                qubit = np.random.randint(0, num_qubits)
                operations.append({'gate': gate, 'qubits': [qubit]})
            
            elif gate in ['rx', 'ry', 'rz']:
                # Parameterized single qubit gate
                qubit = np.random.randint(0, num_qubits)
                angle = np.random.uniform(0, 2 * np.pi)
                operations.append({
                    'gate': gate,
                    'qubits': [qubit],
                    'parameters': [angle]
                })
            
            elif gate == 'cx':
                # Two qubit gate
                control = np.random.randint(0, num_qubits)
                target = np.random.randint(0, num_qubits)
                while target == control:
                    target = np.random.randint(0, num_qubits)
                operations.append({
                    'gate': gate,
                    'qubits': [control, target]
                })
        
        operations.append({'gate': 'measure_all'})
        
        return {
            'name': f'random_circuit_{num_qubits}q_d{depth}',
            'description': f'Random circuit with {num_qubits} qubits and depth {depth}',
            'qubits': num_qubits,
            'classical_bits': num_qubits,
            'operations': operations
        }
    
    @staticmethod
    def deutsch_jozsa_circuit(oracle_type: str = 'constant') -> Dict[str, Any]:
        """Create a Deutsch-Jozsa algorithm circuit fixture."""
        num_qubits = 4  # 3 input qubits + 1 ancilla
        operations = []
        
        # Initialize ancilla qubit in |1⟩
        operations.append({'gate': 'x', 'qubits': [3]})
        
        # Apply Hadamard to all qubits
        for qubit in range(num_qubits):
            operations.append({'gate': 'h', 'qubits': [qubit]})
        
        # Oracle implementation
        if oracle_type == 'constant':
            # Constant function (do nothing)
            pass
        elif oracle_type == 'balanced':
            # Balanced function (flip ancilla based on input)
            operations.append({'gate': 'cx', 'qubits': [0, 3]})
            operations.append({'gate': 'cx', 'qubits': [1, 3]})
        
        # Apply Hadamard to input qubits
        for qubit in range(3):
            operations.append({'gate': 'h', 'qubits': [qubit]})
        
        # Measure input qubits
        operations.append({
            'gate': 'measure',
            'qubits': [0, 1, 2],
            'classical': [0, 1, 2]
        })
        
        expected_constant = {'000': 1.0}
        expected_balanced = {f'{i:03b}': 1/8 for i in range(1, 8)}  # Any state except |000⟩
        
        return {
            'name': f'deutsch_jozsa_{oracle_type}',
            'description': f'Deutsch-Jozsa algorithm with {oracle_type} oracle',
            'qubits': num_qubits,
            'classical_bits': 3,
            'operations': operations,
            'expected_results': {
                'ideal': expected_constant if oracle_type == 'constant' else expected_balanced
            }
        }
    
    @staticmethod
    def quantum_teleportation_circuit() -> Dict[str, Any]:
        """Create a quantum teleportation circuit fixture."""
        operations = [
            # Create Bell pair between qubits 1 and 2
            {'gate': 'h', 'qubits': [1]},
            {'gate': 'cx', 'qubits': [1, 2]},
            
            # Prepare state to teleport on qubit 0 (|+⟩ state)
            {'gate': 'h', 'qubits': [0]},
            
            # Bell measurement on qubits 0 and 1
            {'gate': 'cx', 'qubits': [0, 1]},
            {'gate': 'h', 'qubits': [0]},
            {'gate': 'measure', 'qubits': [0, 1], 'classical': [0, 1]},
            
            # Classical corrections on qubit 2
            {'gate': 'cz', 'control_classical': [0], 'qubits': [2]},
            {'gate': 'cx', 'control_classical': [1], 'qubits': [2]},
            
            # Measure final state
            {'gate': 'measure', 'qubits': [2], 'classical': [2]}
        ]
        
        return {
            'name': 'quantum_teleportation',
            'description': 'Quantum teleportation protocol',
            'qubits': 3,
            'classical_bits': 3,
            'operations': operations,
            'expected_results': {
                'ideal': {
                    '000': 0.25, '001': 0.25, '010': 0.25, '011': 0.25,
                    '100': 0.25, '101': 0.25, '110': 0.25, '111': 0.25
                }
            }
        }


# Pytest fixtures
@pytest.fixture
def bell_state():
    """Provide Bell state circuit fixture."""
    return QuantumCircuitFixtures.bell_state_circuit()


@pytest.fixture
def ghz_state():
    """Provide 3-qubit GHZ state circuit fixture."""
    return QuantumCircuitFixtures.ghz_state_circuit(3)


@pytest.fixture
def qft_circuit():
    """Provide 3-qubit QFT circuit fixture."""
    return QuantumCircuitFixtures.quantum_fourier_transform_circuit(3)


@pytest.fixture
def vqe_ansatz():
    """Provide VQE ansatz circuit fixture."""
    return QuantumCircuitFixtures.vqe_ansatz_circuit(4, 2)


@pytest.fixture
def qaoa_circuit():
    """Provide QAOA circuit fixture."""
    return QuantumCircuitFixtures.qaoa_circuit(4, 1)


@pytest.fixture
def random_circuit():
    """Provide random circuit fixture."""
    return QuantumCircuitFixtures.random_circuit(3, 10)


@pytest.fixture
def deutsch_jozsa_constant():
    """Provide Deutsch-Jozsa constant oracle fixture."""
    return QuantumCircuitFixtures.deutsch_jozsa_circuit('constant')


@pytest.fixture
def deutsch_jozsa_balanced():
    """Provide Deutsch-Jozsa balanced oracle fixture."""
    return QuantumCircuitFixtures.deutsch_jozsa_circuit('balanced')


@pytest.fixture
def teleportation_circuit():
    """Provide quantum teleportation circuit fixture."""
    return QuantumCircuitFixtures.quantum_teleportation_circuit()


@pytest.fixture(params=[
    'bell_state',
    'ghz_state',
    'qft_circuit',
    'random_circuit'
])
def any_circuit(request):
    """Parametrized fixture that provides various circuit types."""
    circuit_generators = {
        'bell_state': QuantumCircuitFixtures.bell_state_circuit,
        'ghz_state': lambda: QuantumCircuitFixtures.ghz_state_circuit(3),
        'qft_circuit': lambda: QuantumCircuitFixtures.quantum_fourier_transform_circuit(3),
        'random_circuit': lambda: QuantumCircuitFixtures.random_circuit(3, 5)
    }
    
    return circuit_generators[request.param]()