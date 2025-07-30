"""
Simple quantum test example for beginners.

This example demonstrates basic quantum testing patterns using the
quantum-devops-ci framework with a simple Bell state preparation.
"""

import pytest
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture

# Try to import Qiskit, but make test work even if not available
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@quantum_fixture
def bell_circuit():
    """
    Fixture providing a Bell state circuit.
    
    Creates the quantum circuit:
    |ψ⟩ = (|00⟩ + |11⟩) / √2
    
    Returns:
        QuantumCircuit: Bell state preparation circuit
    """
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Create superposition on qubit 0
    qc.cx(0, 1)  # Entangle qubits 0 and 1
    qc.measure_all()  # Measure all qubits
    return qc


@quantum_fixture
def ghz_circuit():
    """
    Fixture providing a 3-qubit GHZ state circuit.
    
    Creates the quantum circuit:
    |ψ⟩ = (|000⟩ + |111⟩) / √2
    
    Returns:
        QuantumCircuit: GHZ state preparation circuit
    """
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    qc = QuantumCircuit(3, 3)
    qc.h(0)  # Create superposition on qubit 0
    qc.cx(0, 1)  # Entangle qubits 0 and 1
    qc.cx(1, 2)  # Entangle qubit 1 with qubit 2
    qc.measure_all()  # Measure all qubits
    return qc


class TestBasicQuantumAlgorithms(NoiseAwareTest):
    """
    Basic quantum algorithm tests demonstrating common patterns.
    
    This test class shows how to use the NoiseAwareTest base class
    to write quantum tests that work across different noise conditions.
    """
    
    @pytest.mark.quantum
    def test_bell_state_ideal(self, bell_circuit):
        """
        Test Bell state preparation under ideal conditions.
        
        This test verifies that our Bell state preparation works correctly
        when there's no noise in the quantum system.
        
        Args:
            bell_circuit: Bell state circuit fixture
        """
        # Run the circuit on ideal simulator
        result = self.run_circuit(bell_circuit, shots=1000)
        
        # Get measurement counts
        counts = result.get_counts()
        
        # Bell state should only produce |00⟩ and |11⟩ outcomes
        allowed_states = {'00', '11'}
        for state in counts.keys():
            assert state in allowed_states, f"Unexpected state {state} measured"
        
        # States should be roughly equally distributed
        total_shots = sum(counts.values())
        for state in allowed_states:
            if state in counts:
                probability = counts[state] / total_shots
                # Allow 10% deviation from ideal 50%
                assert 0.4 <= probability <= 0.6, \
                    f"State {state} probability {probability:.3f} not close to 0.5"
    
    @pytest.mark.quantum
    @pytest.mark.slow
    def test_bell_state_with_noise(self, bell_circuit):
        """
        Test Bell state preparation under realistic noise conditions.
        
        This test runs the same Bell state preparation but includes
        noise to simulate real quantum hardware behavior.
        
        Args:
            bell_circuit: Bell state circuit fixture
        """
        # Run with depolarizing noise
        result = self.run_with_noise(
            bell_circuit,
            noise_model="depolarizing_0.01",  # 1% depolarizing noise
            shots=8192  # More shots for statistical significance
        )
        
        # Calculate fidelity (how close we are to ideal Bell state)
        fidelity = self.calculate_bell_fidelity(result)
        
        # With 1% noise, we expect fidelity > 80%
        assert fidelity > 0.8, f"Bell state fidelity {fidelity:.3f} too low under noise"
        
        # Should still only see |00⟩ and |11⟩ states (mostly)
        counts = result.get_counts()
        bell_states = {'00', '11'}
        bell_probability = sum(counts.get(state, 0) for state in bell_states) / sum(counts.values())
        
        # At least 90% of measurements should be in Bell states
        assert bell_probability > 0.9, \
            f"Bell state probability {bell_probability:.3f} too low under noise"
    
    @pytest.mark.quantum
    def test_ghz_state_preparation(self, ghz_circuit):
        """
        Test 3-qubit GHZ state preparation.
        
        This test demonstrates testing multi-qubit entangled states.
        
        Args:
            ghz_circuit: GHZ state circuit fixture
        """
        # Run the circuit
        result = self.run_circuit(ghz_circuit, shots=2000)
        
        # Get measurement counts
        counts = result.get_counts()
        
        # GHZ state should only produce |000⟩ and |111⟩ outcomes
        allowed_states = {'000', '111'}
        total_shots = sum(counts.values())
        allowed_shots = sum(counts.get(state, 0) for state in allowed_states)
        
        # At least 90% should be in allowed states (account for measurement errors)
        allowed_probability = allowed_shots / total_shots
        assert allowed_probability > 0.9, \
            f"GHZ state purity {allowed_probability:.3f} too low"
    
    @pytest.mark.quantum
    def test_noise_sweep(self, bell_circuit):
        """
        Test Bell state fidelity across different noise levels.
        
        This test demonstrates how to systematically test quantum
        algorithms across a range of noise conditions.
        
        Args:
            bell_circuit: Bell state circuit fixture  
        """
        # Test multiple noise levels
        noise_levels = [0.001, 0.01, 0.05]
        fidelities = []
        
        for noise_level in noise_levels:
            result = self.run_with_noise(
                bell_circuit,
                noise_model=f"depolarizing_{noise_level}",
                shots=4096
            )
            
            fidelity = self.calculate_bell_fidelity(result)
            fidelities.append(fidelity)
            
            print(f"Noise level {noise_level}: fidelity = {fidelity:.3f}")
        
        # Fidelity should decrease with increasing noise
        for i in range(len(fidelities) - 1):
            assert fidelities[i] >= fidelities[i+1], \
                "Fidelity should decrease with increasing noise"
        
        # Even with highest noise, should have reasonable fidelity
        assert fidelities[-1] > 0.7, \
            f"Fidelity {fidelities[-1]:.3f} too low even with {noise_levels[-1]} noise"
    
    @pytest.mark.quantum
    @pytest.mark.integration
    def test_error_mitigation_comparison(self, bell_circuit):
        """
        Compare results with and without error mitigation.
        
        This test demonstrates how to validate error mitigation
        techniques in your quantum algorithms.
        
        Args:
            bell_circuit: Bell state circuit fixture
        """
        noise_level = 0.02  # 2% noise
        
        # Run without error mitigation
        raw_result = self.run_with_noise(
            bell_circuit,
            noise_model=f"depolarizing_{noise_level}",
            shots=4096
        )
        raw_fidelity = self.calculate_bell_fidelity(raw_result)
        
        # Run with error mitigation
        mitigated_result = self.run_with_mitigation(
            bell_circuit,
            noise_level=noise_level,
            method="zero_noise_extrapolation",
            shots=4096
        )
        mitigated_fidelity = self.calculate_bell_fidelity(mitigated_result)
        
        print(f"Raw fidelity: {raw_fidelity:.3f}")
        print(f"Mitigated fidelity: {mitigated_fidelity:.3f}")
        print(f"Improvement: {((mitigated_fidelity - raw_fidelity) / raw_fidelity * 100):.1f}%")
        
        # Error mitigation should improve fidelity
        assert mitigated_fidelity >= raw_fidelity, \
            "Error mitigation should not decrease fidelity"
        
        # Should see at least 5% relative improvement
        improvement = (mitigated_fidelity - raw_fidelity) / raw_fidelity
        assert improvement >= 0.05, \
            f"Error mitigation improvement {improvement:.3f} below 5% threshold"


@pytest.mark.quantum
@pytest.mark.parametrize("shots", [100, 1000, 10000])
def test_shot_scaling(shots, bell_circuit):
    """
    Test how measurement accuracy scales with number of shots.
    
    This test demonstrates statistical convergence in quantum measurements.
    
    Args:
        shots: Number of measurement shots to use
        bell_circuit: Bell state circuit fixture
    """
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    # Create a simple test instance
    tester = NoiseAwareTest(default_shots=shots)
    
    # Run Bell state circuit
    result = tester.run_circuit(bell_circuit, shots=shots)
    
    # Calculate fidelity
    fidelity = tester.calculate_bell_fidelity(result)
    
    # With more shots, we expect more accurate results
    # Set minimum fidelity based on shot count
    if shots >= 10000:
        min_fidelity = 0.95
    elif shots >= 1000:
        min_fidelity = 0.90
    else:
        min_fidelity = 0.80
    
    assert fidelity >= min_fidelity, \
        f"Fidelity {fidelity:.3f} below threshold {min_fidelity:.3f} for {shots} shots"


# Performance test example
@pytest.mark.quantum
@pytest.mark.slow
def test_performance_benchmark(bell_circuit):
    """
    Benchmark quantum circuit execution performance.
    
    This test measures execution time and can be used for
    performance regression testing.
    """
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    import time
    
    tester = NoiseAwareTest()
    
    # Warm up
    tester.run_circuit(bell_circuit, shots=100)
    
    # Benchmark execution
    start_time = time.time()
    result = tester.run_circuit(bell_circuit, shots=10000)
    execution_time = time.time() - start_time
    
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Shots per second: {10000/execution_time:.0f}")
    
    # Should complete within reasonable time (varies by system)
    assert execution_time < 30, f"Execution took too long: {execution_time:.2f}s"
    
    # Verify result quality
    fidelity = tester.calculate_bell_fidelity(result)
    assert fidelity > 0.95, f"Benchmark fidelity {fidelity:.3f} below threshold"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])