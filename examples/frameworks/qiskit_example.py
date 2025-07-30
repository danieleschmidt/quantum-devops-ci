"""
Qiskit-specific quantum DevOps CI/CD example.

This example demonstrates how to write quantum tests specifically
for Qiskit circuits with hardware-specific considerations.
"""

import pytest
import numpy as np
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT, QuantumVolume
    from qiskit.quantum_info import Statevector, random_statevector
    from qiskit.providers.aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@quantum_fixture
def variational_circuit():
    """Create a parameterized variational quantum circuit."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    from qiskit.circuit import Parameter
    
    # Create parameterized circuit for VQE
    qc = QuantumCircuit(4)
    
    # Parameters for rotation angles
    theta = [Parameter(f'θ{i}') for i in range(8)]
    
    # Layer 1: Single-qubit rotations
    for i in range(4):
        qc.ry(theta[i], i)
    
    # Layer 2: Entangling gates
    for i in range(3):
        qc.cx(i, i+1)
    
    # Layer 3: More single-qubit rotations
    for i in range(4):
        qc.ry(theta[i+4], i)
    
    # Add measurements
    qc.add_register(ClassicalRegister(4, 'c'))
    qc.measure_all()
    
    return qc


@quantum_fixture
def qft_circuit():
    """Create a Quantum Fourier Transform circuit."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    # Create 4-qubit QFT circuit
    qc = QuantumCircuit(4, 4)
    
    # Apply QFT
    qft = QFT(4)
    qc.compose(qft, inplace=True)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc


@quantum_fixture
def quantum_volume_circuit():
    """Create a quantum volume circuit for benchmarking."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    # Create quantum volume circuit
    qv_circuit = QuantumVolume(4, depth=4, seed=42)
    
    # Add measurements
    qc = QuantumCircuit(4, 4)
    qc.compose(qv_circuit, inplace=True)
    qc.measure_all()
    
    return qc


class TestQiskitSpecificFeatures(NoiseAwareTest):
    """
    Test class demonstrating Qiskit-specific testing patterns.
    
    This class shows how to test Qiskit circuits with specific
    considerations for IBM Quantum hardware and simulators.
    """
    
    @pytest.mark.qiskit
    def test_circuit_transpilation(self, variational_circuit):
        """
        Test circuit transpilation for different backends.
        
        This test ensures circuits can be properly transpiled
        for different quantum hardware architectures.
        """
        # Bind parameters to specific values
        param_values = [np.pi/4 * i for i in range(8)]
        bound_circuit = variational_circuit.bind_parameters(param_values)
        
        # Test transpilation for different coupling maps
        coupling_maps = [
            [(0, 1), (1, 2), (2, 3)],  # Linear
            [(0, 1), (1, 2), (2, 3), (3, 0)],  # Ring
            [(0, 1), (0, 2), (1, 3), (2, 3)]   # Custom
        ]
        
        for i, coupling_map in enumerate(coupling_maps):
            # Transpile circuit
            transpiled = transpile(
                bound_circuit, 
                coupling_map=coupling_map,
                optimization_level=2
            )
            
            # Verify transpilation preserves circuit structure
            assert transpiled.num_qubits == bound_circuit.num_qubits
            assert transpiled.num_clbits == bound_circuit.num_clbits
            
            # Check that only allowed gates are used
            allowed_gates = {'u1', 'u2', 'u3', 'cx', 'measure'}
            for instruction in transpiled.data:
                assert instruction.operation.name in allowed_gates, \
                    f"Unsupported gate {instruction.operation.name} in transpiled circuit"
            
            print(f"Coupling map {i+1}: depth = {transpiled.depth()}, gates = {len(transpiled.data)}")
    
    @pytest.mark.qiskit
    def test_qft_correctness(self, qft_circuit):
        """
        Test Quantum Fourier Transform correctness.
        
        This test verifies that QFT produces expected frequency analysis.
        """
        # Create input state |0001⟩ (binary 1)
        input_circuit = QuantumCircuit(4, 4)
        input_circuit.x(3)  # Set last qubit to |1⟩
        
        # Combine with QFT
        full_circuit = input_circuit.compose(qft_circuit)
        
        # Run simulation
        result = self.run_circuit(full_circuit, shots=8192)
        counts = result.get_counts()
        
        # For input |1⟩, QFT should produce uniform superposition
        # with specific phase relationships
        expected_states = [f'{i:04b}' for i in range(16)]
        
        # Check that we see all computational basis states
        measured_states = set(counts.keys())
        overlap = len(measured_states.intersection(set(expected_states)))
        
        # Should see at least 12 out of 16 possible states
        assert overlap >= 12, f"Only {overlap}/16 expected states observed"
        
        # Check for roughly uniform distribution (within statistical bounds)
        total_shots = sum(counts.values())
        expected_prob = 1.0 / 16  # Uniform distribution
        
        for state in measured_states:
            if counts[state] > 100:  # Only check states with significant counts
                observed_prob = counts[state] / total_shots
                # Allow 3-sigma deviation
                sigma = np.sqrt(expected_prob * (1 - expected_prob) / total_shots)
                assert abs(observed_prob - expected_prob) < 3 * sigma, \
                    f"State {state} probability {observed_prob:.3f} deviates too much from {expected_prob:.3f}"
    
    @pytest.mark.qiskit
    @pytest.mark.slow
    def test_quantum_volume_benchmark(self, quantum_volume_circuit):
        """
        Test quantum volume circuit as performance benchmark.
        
        Quantum volume is a protocol for measuring quantum computer
        performance that accounts for gate errors, connectivity, and more.
        """
        # Run quantum volume circuit
        result = self.run_circuit(quantum_volume_circuit, shots=8192)
        counts = result.get_counts()
        
        # Calculate heavy output probability (quantum volume metric)
        heavy_outputs = self._calculate_heavy_outputs(quantum_volume_circuit, counts)
        heavy_output_prob = sum(counts.get(output, 0) for output in heavy_outputs) / sum(counts.values())
        
        print(f"Heavy output probability: {heavy_output_prob:.3f}")
        
        # For ideal quantum volume, heavy output probability should be > 2/3
        # But with noise and limited depth, we expect lower values
        assert heavy_output_prob > 0.4, f"Heavy output probability {heavy_output_prob:.3f} too low"
        
        # Check circuit depth is reasonable for hardware
        circuit_depth = quantum_volume_circuit.depth()
        assert circuit_depth <= 20, f"Circuit depth {circuit_depth} too high for near-term hardware"
    
    @pytest.mark.qiskit
    def test_noise_model_validation(self, variational_circuit):
        """
        Test different noise models for realistic simulation.
        
        This test validates that circuits behave differently under
        various noise conditions, important for hardware preparation.
        """
        # Bind parameters
        param_values = [np.pi/8 * i for i in range(8)]
        bound_circuit = variational_circuit.bind_parameters(param_values)
        
        # Test different noise scenarios
        noise_scenarios = [
            ("ideal", None),
            ("low_noise", "depolarizing_0.001"),
            ("medium_noise", "depolarizing_0.01"), 
            ("high_noise", "depolarizing_0.05")
        ]
        
        fidelities = []
        
        for scenario_name, noise_model in noise_scenarios:
            if noise_model is None:
                result = self.run_circuit(bound_circuit, shots=4096)
            else:
                result = self.run_with_noise(bound_circuit, noise_model, shots=4096)
            
            # Calculate state fidelity (simplified)
            fidelity = self._calculate_circuit_fidelity(result)
            fidelities.append(fidelity)
            
            print(f"{scenario_name}: fidelity = {fidelity:.3f}")
        
        # Fidelity should decrease with increasing noise
        for i in range(len(fidelities) - 1):
            assert fidelities[i] >= fidelities[i+1] - 0.05, \
                f"Fidelity should decrease with noise: {fidelities[i]:.3f} -> {fidelities[i+1]:.3f}"
        
        # Even high noise should maintain some coherence
        assert fidelities[-1] > 0.3, f"High noise fidelity {fidelities[-1]:.3f} too low"
    
    @pytest.mark.qiskit
    @pytest.mark.integration
    def test_hardware_compatibility(self, variational_circuit):
        """
        Test circuit compatibility with real IBM Quantum hardware.
        
        This test checks hardware constraints without requiring
        actual hardware access.
        """
        # Bind parameters
        param_values = [np.pi/6 * i for i in range(8)]
        bound_circuit = variational_circuit.bind_parameters(param_values)
        
        # Test hardware compatibility constraints
        
        # 1. Circuit depth should be reasonable for NISQ devices
        circuit_depth = bound_circuit.depth()
        assert circuit_depth <= 50, f"Circuit depth {circuit_depth} too high for NISQ hardware"
        
        # 2. Number of two-qubit gates should be limited
        two_qubit_gates = sum(1 for instr in bound_circuit.data if len(instr.qubits) == 2)
        assert two_qubit_gates <= 30, f"Too many two-qubit gates: {two_qubit_gates}"
        
        # 3. Test transpilation for IBM hardware topology
        # Simulate heavy-hex topology (simplified)
        coupling_map = [
            (0, 1), (1, 2), (2, 3),
            (1, 4), (4, 7), (7, 10),
            (2, 5), (5, 8), (8, 11),
            (3, 6), (6, 9), (9, 12)
        ][:10]  # Truncated for 4-qubit circuit
        
        try:
            transpiled = transpile(
                bound_circuit,
                coupling_map=coupling_map,
                optimization_level=3,
                seed_transpiler=42
            )
            
            # Verify successful transpilation
            assert transpiled.depth() > 0, "Transpilation failed"
            
            # Check gate count increase is reasonable
            original_gates = len(bound_circuit.data)
            transpiled_gates = len(transpiled.data)
            gate_overhead = (transpiled_gates - original_gates) / original_gates
            
            assert gate_overhead < 2.0, f"Gate overhead {gate_overhead:.2f} too high"
            
            print(f"Transpilation: {original_gates} -> {transpiled_gates} gates ({gate_overhead:.1%} increase)")
            
        except Exception as e:
            pytest.fail(f"Circuit transpilation failed: {e}")
    
    @pytest.mark.qiskit
    @pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
    def test_optimization_levels(self, variational_circuit, optimization_level):
        """
        Test different transpilation optimization levels.
        
        This test ensures circuits can be optimized at different levels
        and validates the trade-offs between compilation time and quality.
        """
        # Bind parameters
        param_values = [np.pi/7 * i for i in range(8)]
        bound_circuit = variational_circuit.bind_parameters(param_values)
        
        # Transpile with specific optimization level
        import time
        start_time = time.time()
        
        transpiled = transpile(
            bound_circuit,
            optimization_level=optimization_level,
            seed_transpiler=42
        )
        
        compilation_time = time.time() - start_time
        
        # Verify transpilation worked
        assert transpiled.depth() > 0, f"Optimization level {optimization_level} failed"
        
        # Record metrics
        original_depth = bound_circuit.depth()
        optimized_depth = transpiled.depth()
        depth_reduction = (original_depth - optimized_depth) / original_depth
        
        print(f"Opt level {optimization_level}: "
              f"depth {original_depth} -> {optimized_depth} "
              f"({depth_reduction:.1%} reduction), "
              f"time {compilation_time:.3f}s")
        
        # Higher optimization levels should generally produce better results
        if optimization_level >= 2:
            assert optimized_depth <= original_depth, \
                "Higher optimization should not increase depth"
        
        # Compilation time should be reasonable
        assert compilation_time < 10.0, f"Compilation took too long: {compilation_time:.2f}s"
    
    def _calculate_heavy_outputs(self, circuit, counts):
        """Calculate heavy outputs for quantum volume protocol."""
        # Simplified heavy output calculation
        # In practice, this would use the ideal probability distribution
        total_counts = sum(counts.values())
        median_count = total_counts // (2 ** circuit.num_qubits)
        
        heavy_outputs = [state for state, count in counts.items() 
                        if count >= median_count]
        
        return heavy_outputs
    
    def _calculate_circuit_fidelity(self, result):
        """Calculate a simplified circuit fidelity metric."""
        counts = result.get_counts()
        
        # For demonstration, use entropy as a fidelity proxy
        total_shots = sum(counts.values())
        probabilities = [count / total_shots for count in counts.values()]
        
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize to [0, 1] range (higher entropy = lower fidelity)
        max_entropy = np.log2(len(counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Convert to fidelity-like metric
        fidelity = 1 - normalized_entropy
        
        return max(0, min(1, fidelity))


if __name__ == "__main__":
    # Run tests with Qiskit-specific markers
    pytest.main([__file__, "-v", "-m", "qiskit", "--tb=short"])