"""
Example quantum algorithm tests demonstrating testing patterns.

This module shows how to write comprehensive tests for quantum algorithms
using the quantum-devops-ci testing framework with various noise conditions
and hardware compatibility checks.
"""

import pytest
import numpy as np
from quantum_devops_ci.testing import NoiseAwareTest, quantum_fixture

# Try importing quantum frameworks
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import TwoLocal, EfficientSU2
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@quantum_fixture
def vqe_ansatz():
    """Variational Quantum Eigensolver ansatz circuit."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    # Create EfficientSU2 ansatz
    ansatz = EfficientSU2(4, reps=2, entanglement='linear')
    
    # Add measurements
    qc = QuantumCircuit(4, 4)
    qc.compose(ansatz, inplace=True)
    qc.measure_all()
    
    return qc


@quantum_fixture  
def qaoa_circuit():
    """Quantum Approximate Optimization Algorithm circuit."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    from qiskit.circuit import Parameter
    
    # QAOA circuit for Max-Cut on 4-node graph
    qc = QuantumCircuit(4, 4)
    
    # Parameters
    beta = Parameter('β')
    gamma = Parameter('γ')
    
    # Initial superposition
    qc.h(range(4))
    
    # Problem Hamiltonian (Max-Cut edges)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for edge in edges:
        qc.rzz(2 * gamma, edge[0], edge[1])
    
    # Mixer Hamiltonian
    for qubit in range(4):
        qc.rx(2 * beta, qubit)
    
    qc.measure_all()
    return qc


@quantum_fixture
def qml_feature_map():
    """Quantum machine learning feature map circuit."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")
    
    from qiskit.circuit import Parameter
    
    # ZZFeatureMap-like circuit
    qc = QuantumCircuit(4, 4)
    
    # Feature parameters
    features = [Parameter(f'x{i}') for i in range(4)]
    
    # First layer: encode features
    for i, feature in enumerate(features):
        qc.ry(feature, i)
    
    # Second layer: entangling gates with feature products
    for i in range(3):
        qc.rzz(features[i] * features[i+1], i, i+1)
    
    qc.measure_all()
    return qc


class TestQuantumAlgorithmSuite(NoiseAwareTest):
    """
    Comprehensive test suite for quantum algorithms.
    
    This test class demonstrates industry-standard testing patterns
    for quantum algorithms including noise analysis, hardware compatibility,
    and performance benchmarking.
    """
    
    @pytest.mark.quantum
    @pytest.mark.unit
    def test_vqe_ansatz_structure(self, vqe_ansatz):
        """
        Test VQE ansatz circuit structure and properties.
        
        This test validates the structure of a VQE ansatz without
        execution, checking circuit depth, gate counts, and connectivity.
        """
        # Check circuit dimensions
        assert vqe_ansatz.num_qubits == 4, "VQE ansatz should use 4 qubits"
        assert vqe_ansatz.num_clbits == 4, "VQE ansatz should have 4 classical bits"
        
        # Check circuit depth is reasonable for VQE
        depth = vqe_ansatz.depth()
        assert 10 <= depth <= 50, f"VQE circuit depth {depth} outside expected range [10, 50]"
        
        # Check gate composition
        gate_counts = {}
        for instruction in vqe_ansatz.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        # Should have rotation gates for parameterization
        rotation_gates = sum(gate_counts.get(gate, 0) for gate in ['ry', 'rz', 'rx'])
        assert rotation_gates >= 8, f"VQE should have sufficient rotation gates, found {rotation_gates}"
        
        # Should have entangling gates
        entangling_gates = gate_counts.get('cx', 0)
        assert entangling_gates >= 4, f"VQE should have entangling gates, found {entangling_gates}"
        
        print(f"VQE circuit: depth={depth}, gates={gate_counts}")
    
    @pytest.mark.quantum
    @pytest.mark.integration
    def test_vqe_energy_optimization(self, vqe_ansatz):
        """
        Test VQE energy optimization convergence.
        
        This test simulates a VQE optimization loop to ensure
        the ansatz can find ground state energies.
        """
        # Bind parameters to multiple configurations
        num_parameters = vqe_ansatz.num_parameters
        
        # Test multiple parameter configurations
        energies = []
        for trial in range(5):
            # Random parameter values
            params = np.random.uniform(-np.pi, np.pi, num_parameters)
            bound_circuit = vqe_ansatz.bind_parameters(params)
            
            # Run circuit and calculate "energy" (simplified)
            result = self.run_circuit(bound_circuit, shots=4096)
            energy = self._calculate_mock_energy(result)
            energies.append(energy)
            
            print(f"Trial {trial+1}: energy = {energy:.4f}")
        
        # Check that we get a range of energies (optimization space exists)
        energy_range = max(energies) - min(energies)
        assert energy_range > 0.1, f"Energy range {energy_range:.4f} too small for optimization"
        
        # Best energy should be reasonable
        best_energy = min(energies)
        assert -2.0 <= best_energy <= 0.0, f"Best energy {best_energy:.4f} outside expected range"
    
    @pytest.mark.quantum
    @pytest.mark.slow
    def test_vqe_noise_robustness(self, vqe_ansatz):
        """
        Test VQE robustness under different noise conditions.
        
        This test evaluates how VQE performance degrades with
        increasing noise levels, important for NISQ devices.
        """
        # Bind to fixed parameters for consistent comparison
        num_parameters = vqe_ansatz.num_parameters
        fixed_params = np.ones(num_parameters) * np.pi/4
        bound_circuit = vqe_ansatz.bind_parameters(fixed_params)
        
        # Test different noise levels
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02]
        noise_results = []
        
        for noise_level in noise_levels:
            if noise_level == 0.0:
                result = self.run_circuit(bound_circuit, shots=8192)
            else:
                result = self.run_with_noise(
                    bound_circuit,
                    noise_model=f"depolarizing_{noise_level}",
                    shots=8192
                )
            
            energy = self._calculate_mock_energy(result)
            variance = self._calculate_energy_variance(result)
            
            noise_results.append({
                'noise_level': noise_level,
                'energy': energy,
                'variance': variance
            })
            
            print(f"Noise {noise_level}: energy={energy:.4f}, variance={variance:.4f}")
        
        # Verify noise effects
        ideal_energy = noise_results[0]['energy']
        
        for i, result in enumerate(noise_results[1:], 1):
            noise_level = result['noise_level']
            noisy_energy = result['energy']
            variance = result['variance']
            
            # Energy should shift with noise
            energy_shift = abs(noisy_energy - ideal_energy)
            expected_shift = noise_level * 2  # Rough estimate
            
            assert energy_shift <= expected_shift * 2, \
                f"Energy shift {energy_shift:.4f} too large for noise level {noise_level}"
            
            # Variance should increase with noise
            if i > 1:
                prev_variance = noise_results[i-1]['variance']
                assert variance >= prev_variance * 0.8, \
                    "Variance should increase with noise"
    
    @pytest.mark.quantum
    @pytest.mark.unit
    def test_qaoa_parameter_structure(self, qaoa_circuit):
        """
        Test QAOA circuit parameter structure.
        
        This test validates that QAOA circuits have the correct
        parameter structure for optimization.
        """
        # Check parameters
        parameters = qaoa_circuit.parameters
        param_names = [p.name for p in parameters]
        
        assert 'β' in param_names, "QAOA should have β (mixer) parameter"
        assert 'γ' in param_names, "QAOA should have γ (problem) parameter"
        
        # Bind parameters and check structure
        bound_circuit = qaoa_circuit.bind_parameters([np.pi/4, np.pi/8])
        
        # Should have appropriate depth for QAOA
        depth = bound_circuit.depth()
        assert 5 <= depth <= 25, f"QAOA depth {depth} outside expected range"
        
        # Should have ZZ gates for problem Hamiltonian
        gate_counts = {}
        for instruction in bound_circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        assert gate_counts.get('rzz', 0) >= 4, "QAOA should have ZZ rotation gates"
        assert gate_counts.get('rx', 0) >= 4, "QAOA should have X rotation gates"
        
        print(f"QAOA circuit: parameters={len(parameters)}, depth={depth}")
    
    @pytest.mark.quantum
    @pytest.mark.integration
    def test_qaoa_optimization_landscape(self, qaoa_circuit):
        """
        Test QAOA optimization landscape exploration.
        
        This test explores the QAOA parameter space to ensure
        optimization can find good solutions.
        """
        # Sample parameter space
        beta_values = np.linspace(0, np.pi, 5)
        gamma_values = np.linspace(0, np.pi, 5)
        
        best_cost = float('inf')
        cost_landscape = []
        
        for beta in beta_values:
            for gamma in gamma_values:
                # Bind parameters
                bound_circuit = qaoa_circuit.bind_parameters([beta, gamma])
                
                # Run circuit
                result = self.run_circuit(bound_circuit, shots=2048)
                
                # Calculate cost (negative of objective for Max-Cut)
                cost = self._calculate_maxcut_cost(result)
                cost_landscape.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = (beta, gamma)
        
        print(f"QAOA landscape: best_cost={best_cost:.4f} at β={best_params[0]:.3f}, γ={best_params[1]:.3f}")
        
        # Should find reasonable solutions
        # For 4-node complete graph, max cut value is 4
        assert best_cost <= 2.0, f"QAOA best cost {best_cost:.4f} should be reasonable"
        
        # Should see variation in landscape
        cost_std = np.std(cost_landscape)
        assert cost_std > 0.1, f"Cost landscape too flat (std={cost_std:.4f})"
    
    @pytest.mark.quantum
    @pytest.mark.unit
    def test_qml_feature_encoding(self, qml_feature_map):
        """
        Test quantum machine learning feature encoding.
        
        This test validates that QML circuits properly encode
        classical data into quantum states.
        """
        # Test with different feature vectors
        test_features = [
            [0.0, 0.0, 0.0, 0.0],  # All zeros
            [np.pi, np.pi, np.pi, np.pi],  # All π
            [np.pi/2, -np.pi/2, np.pi/4, -np.pi/4],  # Mixed
        ]
        
        state_overlaps = []
        
        for i, features in enumerate(test_features):
            # Bind features to circuit
            bound_circuit = qml_feature_map.bind_parameters(features)
            
            # Run circuit
            result = self.run_circuit(bound_circuit, shots=4096)
            
            # Calculate state "signature" (simplified)
            signature = self._calculate_state_signature(result)
            state_overlaps.append(signature)
            
            print(f"Features {i+1}: signature={signature:.4f}")
        
        # Different feature vectors should produce different signatures
        for i in range(len(state_overlaps)):
            for j in range(i+1, len(state_overlaps)):
                overlap = abs(state_overlaps[i] - state_overlaps[j])
                assert overlap > 0.1, \
                    f"Feature vectors {i} and {j} produce too similar states (overlap={overlap:.4f})"
    
    @pytest.mark.quantum
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_algorithm_performance_comparison(self, vqe_ansatz, qaoa_circuit):
        """
        Benchmark performance comparison between algorithms.
        
        This test compares execution time and resource usage
        across different quantum algorithms.
        """
        import time
        
        algorithms = {
            'VQE': vqe_ansatz.bind_parameters(np.ones(vqe_ansatz.num_parameters) * np.pi/6),
            'QAOA': qaoa_circuit.bind_parameters([np.pi/4, np.pi/8])
        }
        
        performance_metrics = {}
        
        for alg_name, circuit in algorithms.items():
            # Measure execution time
            start_time = time.time()
            
            # Run multiple trials for statistical significance
            trial_results = []
            for trial in range(3):
                result = self.run_circuit(circuit, shots=4096)
                trial_results.append(result)
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            avg_fidelity = np.mean([self._calculate_circuit_fidelity(r) for r in trial_results])
            circuit_depth = circuit.depth()
            gate_count = len(circuit.data)
            
            performance_metrics[alg_name] = {
                'execution_time': execution_time,
                'avg_fidelity': avg_fidelity,
                'circuit_depth': circuit_depth,
                'gate_count': gate_count,
                'shots_per_second': (3 * 4096) / execution_time
            }
            
            print(f"{alg_name}: time={execution_time:.2f}s, fidelity={avg_fidelity:.3f}, "
                  f"depth={circuit_depth}, gates={gate_count}")
        
        # Performance assertions
        for alg_name, metrics in performance_metrics.items():
            # Should complete in reasonable time
            assert metrics['execution_time'] < 60, \
                f"{alg_name} took too long: {metrics['execution_time']:.2f}s"
            
            # Should maintain reasonable fidelity
            assert metrics['avg_fidelity'] > 0.7, \
                f"{alg_name} fidelity too low: {metrics['avg_fidelity']:.3f}"
            
            # Should have reasonable throughput
            assert metrics['shots_per_second'] > 100, \
                f"{alg_name} throughput too low: {metrics['shots_per_second']:.0f} shots/s"
    
    # Helper methods for test calculations
    def _calculate_mock_energy(self, result):
        """Calculate mock energy expectation value."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Simple energy model: Hamming weight
        energy = 0.0
        for state, count in counts.items():
            hamming_weight = sum(int(bit) for bit in state)
            # Energy decreases with fewer |1⟩ states
            state_energy = -1.0 + 0.5 * hamming_weight
            energy += state_energy * count / total_shots
        
        return energy
    
    def _calculate_energy_variance(self, result):
        """Calculate energy measurement variance."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Calculate mean energy
        mean_energy = self._calculate_mock_energy(result)
        
        # Calculate variance
        variance = 0.0
        for state, count in counts.items():
            hamming_weight = sum(int(bit) for bit in state)
            state_energy = -1.0 + 0.5 * hamming_weight
            variance += (state_energy - mean_energy)**2 * count / total_shots
        
        return variance
    
    def _calculate_maxcut_cost(self, result):
        """Calculate Max-Cut cost function."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Max-Cut edges for 4-node graph
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        cost = 0.0
        for state, count in counts.items():
            # Count cut edges
            cut_edges = 0
            for i, j in edges:
                if state[i] != state[j]:  # Edge is cut
                    cut_edges += 1
            
            # Cost is negative cut count (we want to maximize cuts)
            state_cost = -cut_edges
            cost += state_cost * count / total_shots
        
        return cost
    
    def _calculate_state_signature(self, result):
        """Calculate a signature for quantum state comparison."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Calculate entropy as signature
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total_shots
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_circuit_fidelity(self, result):
        """Calculate simplified circuit fidelity metric."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Use state purity as fidelity proxy
        purity = sum((count / total_shots)**2 for count in counts.values())
        
        # Convert to fidelity-like measure
        num_states = len(counts)
        max_purity = 1.0
        min_purity = 1.0 / num_states
        
        if max_purity > min_purity:
            normalized_purity = (purity - min_purity) / (max_purity - min_purity)
        else:
            normalized_purity = 1.0
        
        return normalized_purity


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])