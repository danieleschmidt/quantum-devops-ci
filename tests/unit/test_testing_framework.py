"""
Unit tests for the noise-aware testing framework.

Tests the core functionality of NoiseAwareTest class and related
quantum testing utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from quantum_devops_ci.testing import (
    NoiseAwareTest, 
    TestResult, 
    HardwareCompatibilityTest,
    quantum_fixture
)


class TestNoiseAwareTest:
    """Test cases for NoiseAwareTest class."""
    
    def test_initialization(self):
        """Test NoiseAwareTest initialization."""
        tester = NoiseAwareTest(default_shots=500, timeout_seconds=120)
        
        assert tester.default_shots == 500
        assert tester.timeout_seconds == 120
        assert isinstance(tester.backend_cache, dict)
    
    def test_run_circuit_with_mock(self, noise_aware_tester, mock_quantum_circuit):
        """Test circuit execution with mock circuit."""
        # Mock the framework detection and execution
        with patch.object(noise_aware_tester, '_run_qiskit_circuit') as mock_run:
            mock_result = TestResult(
                counts={'00': 450, '11': 550},
                shots=1000,
                execution_time=1.2,
                backend_name='qasm_simulator'
            )
            mock_run.return_value = mock_result
            
            # Add measure attribute to mock circuit
            mock_quantum_circuit.measure = Mock()
            
            result = noise_aware_tester.run_circuit(mock_quantum_circuit, shots=1000)
            
            assert result.shots == 1000
            assert result.backend_name == 'qasm_simulator'
            assert '00' in result.counts
            assert '11' in result.counts
            mock_run.assert_called_once()
    
    def test_run_with_noise_mock(self, noise_aware_tester, mock_quantum_circuit):
        """Test noisy circuit execution with mock."""
        with patch.object(noise_aware_tester, '_run_qiskit_noisy') as mock_run:
            mock_result = TestResult(
                counts={'00': 400, '01': 50, '10': 60, '11': 490},
                shots=1000,
                execution_time=1.5,
                backend_name='noisy_qasm_simulator',
                noise_model='depolarizing_0.01'
            )
            mock_run.return_value = mock_result
            
            # Add measure attribute
            mock_quantum_circuit.measure = Mock()
            
            result = noise_aware_tester.run_with_noise(
                mock_quantum_circuit, 
                'depolarizing_0.01',
                shots=1000
            )
            
            assert result.noise_model == 'depolarizing_0.01'
            assert len(result.counts) == 4  # More noise = more states
            mock_run.assert_called_once()
    
    def test_run_with_noise_sweep(self, noise_aware_tester, mock_quantum_circuit):
        """Test noise sweep functionality."""
        with patch.object(noise_aware_tester, 'run_with_noise') as mock_run:
            # Mock different results for different noise levels
            def side_effect(circuit, noise_model, shots, **kwargs):
                noise_level = float(noise_model.split('_')[1])
                return TestResult(
                    counts={'00': int(500 - noise_level * 1000), '11': int(500 + noise_level * 1000)},
                    shots=shots,
                    execution_time=1.0,
                    backend_name='test_backend'
                )
            
            mock_run.side_effect = side_effect
            mock_quantum_circuit.measure = Mock()
            
            results = noise_aware_tester.run_with_noise_sweep(
                mock_quantum_circuit,
                noise_levels=[0.01, 0.05, 0.1],
                shots=1000
            )
            
            assert len(results) == 3
            assert 0.01 in results
            assert 0.05 in results
            assert 0.1 in results
            
            # Higher noise should affect results
            assert results[0.01].counts['00'] > results[0.1].counts['00']
    
    def test_calculate_bell_fidelity(self, noise_aware_tester):
        """Test Bell state fidelity calculation."""
        # Perfect Bell state
        perfect_result = TestResult(
            counts={'00': 500, '11': 500},
            shots=1000,
            execution_time=1.0,
            backend_name='test'
        )
        fidelity = noise_aware_tester.calculate_bell_fidelity(perfect_result)
        assert fidelity == 1.0
        
        # Noisy Bell state
        noisy_result = TestResult(
            counts={'00': 400, '01': 100, '10': 100, '11': 400},
            shots=1000,
            execution_time=1.0,
            backend_name='test'
        )
        fidelity = noise_aware_tester.calculate_bell_fidelity(noisy_result)
        assert fidelity == 0.8  # 800/1000
        
        # Empty result
        empty_result = TestResult(
            counts={},
            shots=0,
            execution_time=1.0,
            backend_name='test'
        )
        fidelity = noise_aware_tester.calculate_bell_fidelity(empty_result)
        assert fidelity == 0.0
    
    def test_calculate_state_fidelity(self, noise_aware_tester):
        """Test generic state fidelity calculation."""
        result = TestResult(
            counts={'00': 500, '11': 500},
            shots=1000,
            execution_time=1.0,
            backend_name='test'
        )
        
        bell_fidelity = noise_aware_tester.calculate_state_fidelity(result, 'bell')
        assert bell_fidelity == 1.0
        
        # Test uniform distribution
        uniform_result = TestResult(
            counts={'00': 250, '01': 250, '10': 250, '11': 250},
            shots=1000,
            execution_time=1.0,
            backend_name='test'
        )
        uniform_fidelity = noise_aware_tester.calculate_state_fidelity(uniform_result, 'uniform')
        assert uniform_fidelity > 0.9  # Should be close to perfect uniform
        
        # Test unknown state
        with pytest.raises(ValueError):
            noise_aware_tester.calculate_state_fidelity(result, 'unknown_state')
    
    def test_run_with_mitigation(self, noise_aware_tester, mock_quantum_circuit):
        """Test error mitigation functionality."""
        with patch.object(noise_aware_tester, 'run_with_noise') as mock_run:
            mock_result = TestResult(
                counts={'00': 450, '11': 550},
                shots=1000,
                execution_time=1.0,
                backend_name='test'
            )
            mock_run.return_value = mock_result
            
            mock_quantum_circuit.measure = Mock()
            
            # Test mitigation (currently just runs with noise)
            result = noise_aware_tester.run_with_mitigation(
                mock_quantum_circuit,
                noise_level=0.01,
                method="zero_noise_extrapolation"
            )
            
            assert result.shots == 1000
            mock_run.assert_called_once()
    
    @pytest.mark.skipif(True, reason="Requires Qiskit installation")
    def test_qiskit_circuit_execution(self, noise_aware_tester, bell_circuit_qiskit):
        """Test actual Qiskit circuit execution (if available)."""
        # This test would run if Qiskit is available
        result = noise_aware_tester.run_circuit(bell_circuit_qiskit, shots=100)
        
        assert result.shots == 100
        assert result.backend_name in ['qasm_simulator', 'aer_qasm_simulator']
        assert isinstance(result.counts, dict)
        assert result.execution_time > 0
    
    def test_framework_not_available(self, noise_aware_tester):
        """Test behavior when quantum framework is not available."""
        # Create a mock circuit that doesn't have expected attributes
        mock_circuit = Mock()
        # Remove framework-identifying attributes
        del mock_circuit.measure
        del mock_circuit.moments
        
        with pytest.raises(NotImplementedError):
            noise_aware_tester.run_circuit(mock_circuit)


class TestTestResult:
    """Test cases for TestResult class."""
    
    def test_test_result_creation(self):
        """Test TestResult creation and methods."""
        counts = {'00': 480, '01': 20, '10': 30, '11': 470}
        result = TestResult(
            counts=counts,
            shots=1000,
            execution_time=1.5,
            backend_name='test_backend',
            noise_model='test_noise',
            metadata={'test': 'data'}
        )
        
        assert result.get_counts() == counts
        assert result.shots == 1000
        assert result.backend_name == 'test_backend'
        assert result.noise_model == 'test_noise'
        assert result.metadata == {'test': 'data'}
    
    def test_get_probabilities(self):
        """Test probability calculation."""
        result = TestResult(
            counts={'00': 250, '01': 250, '10': 250, '11': 250},
            shots=1000,
            execution_time=1.0,
            backend_name='test'
        )
        
        probabilities = result.get_probabilities()
        
        assert len(probabilities) == 4
        for prob in probabilities.values():
            assert prob == 0.25
        assert sum(probabilities.values()) == 1.0


class TestHardwareCompatibilityTest:
    """Test cases for HardwareCompatibilityTest class."""
    
    def test_initialization(self):
        """Test HardwareCompatibilityTest initialization."""
        tester = HardwareCompatibilityTest()
        
        assert isinstance(tester, NoiseAwareTest)
        assert hasattr(tester, 'get_available_backends')
    
    def test_get_available_backends(self):
        """Test getting available backends."""
        tester = HardwareCompatibilityTest()
        backends = tester.get_available_backends()
        
        assert isinstance(backends, list)
        assert 'qasm_simulator' in backends
    
    def test_get_native_gates(self):
        """Test getting native gate sets."""
        tester = HardwareCompatibilityTest()
        
        # Test simulator gates
        sim_gates = tester.get_native_gates('qasm_simulator')
        assert isinstance(sim_gates, list)
        assert 'cx' in sim_gates
        
        # Test statevector simulator
        sv_gates = tester.get_native_gates('statevector_simulator')
        assert isinstance(sv_gates, list)
        
        # Test unknown backend (should return default)
        unknown_gates = tester.get_native_gates('unknown_backend')
        assert isinstance(unknown_gates, list)
    
    def test_circuits_equivalent(self):
        """Test circuit equivalence checking."""
        tester = HardwareCompatibilityTest()
        
        # Create mock circuits
        circuit1 = Mock()
        circuit1.num_qubits = 2
        
        circuit2 = Mock()
        circuit2.num_qubits = 2
        
        circuit3 = Mock()
        circuit3.num_qubits = 3
        
        # Same number of qubits should be equivalent (simplified check)
        assert tester.circuits_equivalent(circuit1, circuit2)
        
        # Different number of qubits should not be equivalent
        assert not tester.circuits_equivalent(circuit1, circuit3)


class TestQuantumFixture:
    """Test cases for quantum_fixture decorator."""
    
    def test_quantum_fixture_with_pytest(self):
        """Test quantum_fixture when pytest is available."""
        @quantum_fixture
        def sample_circuit():
            return "mock_circuit"
        
        # Check that the function is properly decorated
        # (Actual pytest fixture behavior would be tested in integration tests)
        assert callable(sample_circuit)
    
    def test_quantum_fixture_without_pytest(self):
        """Test quantum_fixture when pytest is not available."""
        # Mock pytest not being available
        with patch('quantum_devops_ci.testing.PYTEST_AVAILABLE', False):
            from quantum_devops_ci.testing import quantum_fixture
            
            @quantum_fixture
            def sample_circuit():
                return "mock_circuit"
            
            # Should return the original function
            assert sample_circuit() == "mock_circuit"


@pytest.mark.performance
class TestPerformance:
    """Performance tests for testing framework."""
    
    def test_test_result_creation_performance(self):
        """Test TestResult creation performance."""
        import time
        
        # Create large counts dictionary
        large_counts = {f'{i:04b}': i for i in range(1000)}
        
        start_time = time.time()
        result = TestResult(
            counts=large_counts,
            shots=sum(large_counts.values()),
            execution_time=1.0,
            backend_name='test'
        )
        creation_time = time.time() - start_time
        
        # Should create quickly even with large data
        assert creation_time < 0.1
        assert len(result.get_counts()) == 1000
    
    def test_fidelity_calculation_performance(self, noise_aware_tester):
        """Test fidelity calculation performance."""
        import time
        
        # Create result with many states
        large_counts = {f'{i:010b}': 100 for i in range(100)}
        large_counts['0000000000'] = 50000  # Add dominant state
        large_counts['1111111111'] = 50000  # Add another dominant state
        
        result = TestResult(
            counts=large_counts,
            shots=sum(large_counts.values()),
            execution_time=1.0,
            backend_name='test'
        )
        
        start_time = time.time()
        fidelity = noise_aware_tester.calculate_bell_fidelity(result)
        calc_time = time.time() - start_time
        
        # Should calculate quickly
        assert calc_time < 0.01
        assert 0.0 <= fidelity <= 1.0