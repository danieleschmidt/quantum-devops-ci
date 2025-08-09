#!/usr/bin/env python3
"""
Simple demonstration of quantum circuit testing without external dependencies.
This shows how the framework would work with actual quantum circuits.
"""

import sys
from pathlib import Path

# Add the source directory to Python path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

def create_mock_bell_circuit():
    """Create a mock Bell circuit representation."""
    class MockQuantumCircuit:
        def __init__(self, name="Bell Circuit"):
            self.name = name
            self.num_qubits = 2
            self.num_clbits = 2
            self.gates = [
                ("h", 0),      # Hadamard on qubit 0
                ("cx", 0, 1),  # CNOT from 0 to 1
                ("measure", 0, 0),  # Measure qubit 0 to classical bit 0
                ("measure", 1, 1)   # Measure qubit 1 to classical bit 1
            ]
        
        def depth(self):
            """Return circuit depth."""
            return 2
        
        def __str__(self):
            return f"MockQuantumCircuit({self.name}, {self.num_qubits} qubits)"
    
    return MockQuantumCircuit()


def simulate_bell_circuit_execution(circuit, shots=1000, noise_level=0.0):
    """Simulate execution of a Bell circuit with optional noise."""
    import random
    
    # Perfect Bell state should give 50% |00⟩ and 50% |11⟩
    # With noise, we get some |01⟩ and |10⟩ states
    
    counts = {'00': 0, '11': 0, '01': 0, '10': 0}
    
    for _ in range(shots):
        # Generate random outcome
        rand = random.random()
        
        if noise_level == 0:
            # Perfect case: only |00⟩ and |11⟩
            if rand < 0.5:
                counts['00'] += 1
            else:
                counts['11'] += 1
        else:
            # Noisy case: distribute errors
            correct_prob = 1 - noise_level
            if rand < correct_prob / 2:
                counts['00'] += 1
            elif rand < correct_prob:
                counts['11'] += 1
            elif rand < correct_prob + noise_level / 2:
                counts['01'] += 1
            else:
                counts['10'] += 1
    
    return counts


def main():
    """Demonstrate quantum circuit testing."""
    print("🔬 Quantum Circuit Testing Demo")
    print("=" * 40)
    
    try:
        from quantum_devops_ci.testing import NoiseAwareTest, TestResult
        
        # Create test framework
        test_runner = NoiseAwareTest(default_shots=1000)
        print("✅ Quantum test framework initialized")
        
        # Create mock circuit
        bell_circuit = create_mock_bell_circuit()
        print(f"✅ Created mock circuit: {bell_circuit}")
        print(f"   - Qubits: {bell_circuit.num_qubits}")
        print(f"   - Depth: {bell_circuit.depth()}")
        print(f"   - Gates: {len(bell_circuit.gates)}")
        
        # Simulate perfect execution
        print("\n🎯 Testing Perfect Bell State:")
        perfect_counts = simulate_bell_circuit_execution(bell_circuit, shots=1000, noise_level=0.0)
        
        perfect_result = TestResult(
            counts=perfect_counts,
            shots=1000,
            execution_time=0.05,
            backend_name="perfect_simulator",
            metadata={'circuit_name': bell_circuit.name}
        )
        
        perfect_fidelity = test_runner.calculate_bell_fidelity(perfect_result)
        print(f"   Counts: {perfect_result.counts}")
        print(f"   Fidelity: {perfect_fidelity:.3f}")
        
        # Simulate noisy execution
        print("\n🌪️  Testing Noisy Bell State (5% noise):")
        noisy_counts = simulate_bell_circuit_execution(bell_circuit, shots=1000, noise_level=0.1)
        
        noisy_result = TestResult(
            counts=noisy_counts,
            shots=1000,
            execution_time=0.08,
            backend_name="noisy_simulator",
            noise_model="depolarizing_0.05",
            metadata={'circuit_name': bell_circuit.name}
        )
        
        noisy_fidelity = test_runner.calculate_bell_fidelity(noisy_result)
        print(f"   Counts: {noisy_result.counts}")
        print(f"   Fidelity: {noisy_fidelity:.3f}")
        print(f"   Fidelity degradation: {perfect_fidelity - noisy_fidelity:.3f}")
        
        # Test noise sweep
        print("\n🌊 Noise Sweep Analysis:")
        noise_levels = [0.01, 0.02, 0.05, 0.10]
        fidelities = []
        
        for noise in noise_levels:
            counts = simulate_bell_circuit_execution(bell_circuit, shots=500, noise_level=noise)
            result = TestResult(
                counts=counts,
                shots=500,
                execution_time=0.03,
                backend_name=f"simulator_noise_{noise}",
                noise_model=f"depolarizing_{noise}"
            )
            
            fidelity = test_runner.calculate_bell_fidelity(result)
            fidelities.append(fidelity)
            print(f"   Noise {noise:4.2f}: Fidelity {fidelity:.3f}")
        
        # Validate that fidelity decreases with noise
        if all(fidelities[i] >= fidelities[i+1] for i in range(len(fidelities)-1)):
            print("✅ Fidelity correctly decreases with increasing noise")
        else:
            print("⚠️  Unexpected fidelity behavior (this can happen due to randomness)")
        
        print("\n🎉 Quantum circuit testing demonstration completed!")
        print("\n📈 This demonstrates:")
        print("   - Quantum circuit representation")
        print("   - Noise-aware testing")
        print("   - Fidelity analysis") 
        print("   - Statistical validation")
        
        print("\n🔮 With real quantum frameworks (Qiskit, Cirq), this would:")
        print("   - Execute on real simulators and quantum hardware")
        print("   - Apply realistic noise models")
        print("   - Generate detailed quantum metrics")
        print("   - Integrate with CI/CD pipelines")
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())