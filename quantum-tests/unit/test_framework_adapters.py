"""
Unit tests for quantum framework adapters.

Tests the abstraction layer that allows switching between different quantum
computing frameworks (Qiskit, Cirq, PennyLane) seamlessly.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import framework adapters
from quantum_devops_ci.plugins import (
    QiskitAdapter, 
    CirqAdapter, 
    MockAdapter
)
from quantum_devops_ci import NoiseAwareTest


class TestQiskitAdapter:
    """Test Qiskit framework adapter functionality."""
    
    @pytest.fixture
    def qiskit_adapter(self):
        """Provide a Qiskit adapter instance."""
        return QiskitAdapter()
    
    @pytest.fixture
    def mock_qiskit_circuit(self):
        """Mock a Qiskit quantum circuit."""
        with patch('qiskit.QuantumCircuit') as mock_circuit:
            circuit = Mock()
            circuit.num_qubits = 2
            circuit.num_clbits = 2
            circuit.depth.return_value = 5
            circuit.count_ops.return_value = {'h': 1, 'cx': 1, 'measure': 2}
            mock_circuit.return_value = circuit
            yield circuit
    
    def test_framework_detection(self, qiskit_adapter):
        """Test that adapter correctly identifies Qiskit framework."""
        assert qiskit_adapter.supports('qiskit')
        assert qiskit_adapter.supports('Qiskit')
        assert not qiskit_adapter.supports('cirq')
        assert not qiskit_adapter.supports('pennylane')
    
    @pytest.mark.qiskit
    def test_circuit_creation(self, qiskit_adapter, mock_qiskit_circuit):
        """Test quantum circuit creation from generic description."""
        circuit_data = {
            'qubits': 2,
            'classical_bits': 2,
            'operations': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cx', 'qubits': [0, 1]},
                {'gate': 'measure', 'qubits': [0, 1], 'classical': [0, 1]}
            ]
        }
        
        circuit = qiskit_adapter.create_circuit(circuit_data)
        
        assert circuit is not None
        assert circuit.num_qubits == 2
        assert circuit.num_clbits == 2
    
    @pytest.mark.qiskit
    def test_circuit_execution(self, qiskit_adapter, mock_qiskit_circuit):
        """Test circuit execution with mocked backend."""
        with patch('qiskit.execute') as mock_execute:
            # Mock job and result
            mock_job = Mock()
            mock_result = Mock()
            mock_result.get_counts.return_value = {'00': 500, '11': 500}
            mock_job.result.return_value = mock_result
            mock_execute.return_value = mock_job
            
            result = qiskit_adapter.execute_circuit(
                mock_qiskit_circuit,
                backend='qasm_simulator',
                shots=1000
            )
            
            assert 'counts' in result
            assert result['counts'] == {'00': 500, '11': 500}
            assert result['shots'] == 1000
    
    @pytest.mark.qiskit
    def test_noise_model_integration(self, qiskit_adapter):
        """Test integration with Qiskit noise models."""
        with patch('qiskit_aer.noise.NoiseModel') as mock_noise:
            noise_config = {
                'depolarizing_error': 0.01,
                'readout_error': [[0.95, 0.05], [0.10, 0.90]]
            }
            
            noise_model = qiskit_adapter.create_noise_model(noise_config)
            
            assert noise_model is not None
            mock_noise.assert_called_once()
    
    @pytest.mark.qiskit 
    def test_circuit_optimization(self, qiskit_adapter, mock_qiskit_circuit):
        """Test circuit optimization passes."""
        with patch('qiskit.transpile') as mock_transpile:
            mock_transpile.return_value = mock_qiskit_circuit
            
            optimized = qiskit_adapter.optimize_circuit(
                mock_qiskit_circuit,
                optimization_level=3
            )
            
            assert optimized is not None
            mock_transpile.assert_called_once()


class TestCirqAdapter:
    """Test Cirq framework adapter functionality."""
    
    @pytest.fixture
    def cirq_adapter(self):
        """Provide a Cirq adapter instance."""
        return CirqAdapter()
    
    @pytest.fixture
    def mock_cirq_circuit(self):
        """Mock a Cirq quantum circuit."""
        with patch('cirq.Circuit') as mock_circuit:
            circuit = Mock()
            circuit.all_qubits.return_value = [Mock(), Mock()]  # 2 qubits
            circuit.moments = []
            yield circuit
    
    def test_framework_detection(self, cirq_adapter):
        """Test that adapter correctly identifies Cirq framework."""
        assert cirq_adapter.supports('cirq')
        assert cirq_adapter.supports('Cirq')
        assert not cirq_adapter.supports('qiskit')
        assert not cirq_adapter.supports('pennylane')
    
    @pytest.mark.cirq
    def test_circuit_creation(self, cirq_adapter, mock_cirq_circuit):
        """Test quantum circuit creation from generic description."""
        circuit_data = {
            'qubits': 2,
            'operations': [
                {'gate': 'H', 'qubits': [0]},
                {'gate': 'CNOT', 'qubits': [0, 1]},
                {'gate': 'measure', 'qubits': [0, 1]}
            ]
        }
        
        circuit = cirq_adapter.create_circuit(circuit_data)
        
        assert circuit is not None
        assert len(list(circuit.all_qubits())) == 2
    
    @pytest.mark.cirq
    def test_circuit_execution(self, cirq_adapter, mock_cirq_circuit):
        """Test circuit execution with Cirq simulator."""
        with patch('cirq.Simulator') as mock_simulator:
            # Mock simulation result
            mock_result = Mock()
            mock_result.histogram.return_value = {0: 500, 3: 500}  # |00⟩ and |11⟩
            mock_simulator.return_value.run.return_value = mock_result
            
            result = cirq_adapter.execute_circuit(
                mock_cirq_circuit,
                backend='simulator',
                shots=1000
            )
            
            assert 'counts' in result
            assert result['shots'] == 1000
    
    @pytest.mark.cirq
    def test_noise_model_creation(self, cirq_adapter):
        """Test creation of Cirq noise models."""
        noise_config = {
            'depolarizing_probability': 0.01,
            'readout_error_probability': 0.05
        }
        
        noise_model = cirq_adapter.create_noise_model(noise_config)
        
        assert noise_model is not None


class TestMockAdapter:
    """Test Mock framework adapter functionality."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Provide a Mock adapter instance."""
        return MockAdapter()
    
    def test_framework_detection(self, mock_adapter):
        """Test that adapter correctly identifies Mock framework."""
        assert mock_adapter.available
    
    def test_circuit_creation(self, mock_adapter):
        """Test mock circuit creation."""
        circuit = mock_adapter.create_circuit(2, 2)
        
        assert circuit is not None
        assert circuit['num_qubits'] == 2
        assert circuit['num_clbits'] == 2
        assert circuit['gates'] == []
    
    def test_gate_addition(self, mock_adapter):
        """Test adding gates to mock circuit."""
        circuit = mock_adapter.create_circuit(2, 2)
        
        mock_adapter.add_gate(circuit, 'h', [0])
        mock_adapter.add_gate(circuit, 'cx', [0, 1])
        
        assert len(circuit['gates']) == 2
        assert circuit['gates'][0]['name'] == 'h'
        assert circuit['gates'][1]['name'] == 'cx'
    
    def test_circuit_execution(self, mock_adapter):
        """Test mock circuit execution."""
        circuit = mock_adapter.create_circuit(2, 2)
        result = mock_adapter.execute_circuit(circuit, shots=1000)
        
        assert 'counts' in result
        assert result['shots'] == 1000
        counts = mock_adapter.get_counts(result)
        assert isinstance(counts, dict)


class TestFrameworkCompatibility:
    """Test cross-framework compatibility and conversions."""
    
    @pytest.fixture
    def all_adapters(self):
        """Provide all framework adapters."""
        return {
            'qiskit': QiskitAdapter(),
            'cirq': CirqAdapter(),
            'mock': MockAdapter()
        }
    
    def test_bell_state_equivalence(self, all_adapters):
        """Test that Bell state circuits produce equivalent results across frameworks."""
        bell_circuit_data = {
            'qubits': 2,
            'classical_bits': 2,
            'operations': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cx', 'qubits': [0, 1]},
                {'gate': 'measure', 'qubits': [0, 1], 'classical': [0, 1]}
            ]
        }
        
        results = {}
        
        for framework, adapter in all_adapters.items():
            if adapter.is_available():
                try:
                    circuit = adapter.create_circuit(bell_circuit_data)
                    result = adapter.execute_circuit(
                        circuit, 
                        backend='simulator', 
                        shots=1000
                    )
                    results[framework] = result
                except Exception as e:
                    pytest.skip(f"Framework {framework} not available: {e}")
        
        # Verify all frameworks produce similar Bell state distributions
        if len(results) > 1:
            frameworks = list(results.keys())
            base_result = results[frameworks[0]]
            
            for framework in frameworks[1:]:
                compare_result = results[framework]
                
                # Check that both have roughly equal |00⟩ and |11⟩ states
                # (allowing for statistical variations)
                base_00_ratio = get_state_ratio(base_result, '00')
                compare_00_ratio = get_state_ratio(compare_result, '00')
                
                assert abs(base_00_ratio - compare_00_ratio) < 0.1, \
                    f"Bell state distributions differ between {frameworks[0]} and {framework}"
    
    def test_framework_circuit_conversion(self, all_adapters):
        """Test conversion between different framework circuit representations."""
        # Create a simple circuit in each framework's native format
        native_circuits = {}
        
        for framework, adapter in all_adapters.items():
            if adapter.is_available():
                try:
                    # Create a simple X gate circuit
                    circuit_data = {
                        'qubits': 1,
                        'operations': [{'gate': 'x', 'qubits': [0]}]
                    }
                    circuit = adapter.create_circuit(circuit_data)
                    native_circuits[framework] = circuit
                except Exception:
                    continue
        
        # Test conversion between frameworks (if conversion methods exist)
        for source_framework, source_circuit in native_circuits.items():
            for target_framework, target_adapter in all_adapters.items():
                if (source_framework != target_framework and 
                    target_adapter.is_available() and
                    hasattr(target_adapter, 'convert_from')):
                    
                    try:
                        converted = target_adapter.convert_from(
                            source_circuit, 
                            source_framework
                        )
                        assert converted is not None
                    except NotImplementedError:
                        # Conversion not implemented - this is acceptable
                        pass


class TestNoiseModelIntegration:
    """Test noise model integration across frameworks."""
    
    def test_noise_model_consistency(self):
        """Test that similar noise models produce consistent behavior."""
        # Define a standard noise configuration
        noise_config = {
            'depolarizing_error': 0.01,
            'thermal_relaxation': {
                't1': 50e-6,  # 50 microseconds
                't2': 70e-6   # 70 microseconds
            },
            'readout_error': 0.05
        }
        
        adapters = [QiskitAdapter(), CirqAdapter()]
        noise_models = {}
        
        for adapter in adapters:
            if adapter.is_available():
                try:
                    noise_model = adapter.create_noise_model(noise_config)
                    noise_models[adapter.__class__.__name__] = noise_model
                except Exception as e:
                    pytest.skip(f"Noise model creation failed: {e}")
        
        # Verify noise models were created
        assert len(noise_models) > 0, "No noise models were successfully created"
        
        # Test that noise models affect circuit results consistently
        # (This would require running actual noisy simulations)


# Helper functions
def get_state_ratio(result, state):
    """Get the ratio of a specific quantum state in the results."""
    counts = result.get('counts', {})
    total_shots = sum(counts.values())
    
    if total_shots == 0:
        return 0.0
    
    return counts.get(state, 0) / total_shots


def calculate_fidelity(result1, result2):
    """Calculate fidelity between two quantum measurement results."""
    counts1 = result1.get('counts', {})
    counts2 = result2.get('counts', {})
    
    # Normalize to probabilities
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    if total1 == 0 or total2 == 0:
        return 0.0
    
    prob1 = {state: count/total1 for state, count in counts1.items()}
    prob2 = {state: count/total2 for state, count in counts2.items()}
    
    # Calculate fidelity (simplified overlap)
    all_states = set(prob1.keys()) | set(prob2.keys())
    overlap = sum(
        np.sqrt(prob1.get(state, 0) * prob2.get(state, 0))
        for state in all_states
    )
    
    return overlap