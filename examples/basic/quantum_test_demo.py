#!/usr/bin/env python3
"""
Simple demonstration of the quantum DevOps CI testing framework.

This example shows how to use the NoiseAwareTest class to run quantum
circuits with different noise levels and collect metrics.
"""

import sys
import warnings
from pathlib import Path

# Add the source directory to Python path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from quantum_devops_ci.testing import NoiseAwareTest, TestResult
    from quantum_devops_ci.database.migrations import run_migrations
    print("✅ Successfully imported quantum_devops_ci modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# Check if Qiskit is available
try:
    import qiskit
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    QISKIT_AVAILABLE = True
    print("✅ Qiskit is available")
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Qiskit not available - install with: pip install qiskit qiskit-aer")


def create_bell_circuit() -> QuantumCircuit:
    """Create a simple Bell state preparation circuit."""
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available")
    
    # Create quantum circuit
    qreg = QuantumRegister(2, 'q')
    creg = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qreg, creg)
    
    # Create Bell state: |00⟩ + |11⟩
    circuit.h(qreg[0])    # Hadamard on qubit 0
    circuit.cx(qreg[0], qreg[1])  # CNOT from qubit 0 to 1
    circuit.measure(qreg, creg)   # Measure both qubits
    
    return circuit


def create_ghz_circuit() -> QuantumCircuit:
    """Create a 3-qubit GHZ state circuit."""
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available")
    
    qreg = QuantumRegister(3, 'q')
    creg = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg, creg)
    
    # Create GHZ state: |000⟩ + |111⟩
    circuit.h(qreg[0])    # Hadamard on qubit 0
    circuit.cx(qreg[0], qreg[1])  # CNOT from qubit 0 to 1
    circuit.cx(qreg[1], qreg[2])  # CNOT from qubit 1 to 2
    circuit.measure(qreg, creg)   # Measure all qubits
    
    return circuit


def demonstrate_basic_testing():
    """Demonstrate basic quantum testing functionality."""
    print("\n🧪 Basic Quantum Testing Demo")
    print("=" * 50)
    
    if not QISKIT_AVAILABLE:
        print("⚠️  Skipping Qiskit tests - Qiskit not available")
        return
    
    # Initialize test framework
    test_runner = NoiseAwareTest(default_shots=1000, timeout_seconds=60)
    
    try:
        # Test Bell state circuit
        print("\n1. Testing Bell State Circuit")
        bell_circuit = create_bell_circuit()
        print(f"   Circuit depth: {bell_circuit.depth()}")
        print(f"   Number of qubits: {bell_circuit.num_qubits}")
        
        # Run noiseless simulation
        print("   Running noiseless simulation...")
        result = test_runner.run_circuit(bell_circuit, shots=1000, backend="qasm_simulator")
        
        print(f"   Execution time: {result.execution_time:.3f}s")
        print(f"   Measurement counts: {result.counts}")
        
        # Calculate fidelity
        fidelity = test_runner.calculate_bell_fidelity(result)
        print(f"   Bell state fidelity: {fidelity:.3f}")
        
        # Test with noise
        print("\n   Running with noise (depolarizing error rate: 0.01)...")
        noisy_result = test_runner.run_with_noise(
            bell_circuit, 
            "depolarizing_0.01", 
            shots=1000
        )
        
        print(f"   Noisy execution time: {noisy_result.execution_time:.3f}s")
        print(f"   Noisy measurement counts: {noisy_result.counts}")
        
        noisy_fidelity = test_runner.calculate_bell_fidelity(noisy_result)
        print(f"   Noisy Bell state fidelity: {noisy_fidelity:.3f}")
        print(f"   Fidelity degradation: {(fidelity - noisy_fidelity):.3f}")
        
        print("   ✅ Bell state test completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Bell state test failed: {e}")
        return False
    
    try:
        # Test GHZ state circuit
        print("\n2. Testing GHZ State Circuit")
        ghz_circuit = create_ghz_circuit()
        print(f"   Circuit depth: {ghz_circuit.depth()}")
        print(f"   Number of qubits: {ghz_circuit.num_qubits}")
        
        # Run simulation
        print("   Running noiseless simulation...")
        result = test_runner.run_circuit(ghz_circuit, shots=1000, backend="qasm_simulator")
        
        print(f"   Execution time: {result.execution_time:.3f}s")
        print(f"   Measurement counts: {result.counts}")
        
        # Check if we got expected GHZ correlations
        total_counts = sum(result.counts.values())
        expected_states = ['000', '111']
        correct_counts = sum(result.counts.get(state, 0) for state in expected_states)
        ghz_fidelity = correct_counts / total_counts if total_counts > 0 else 0
        
        print(f"   GHZ state fidelity: {ghz_fidelity:.3f}")
        print("   ✅ GHZ state test completed successfully!")
        
    except Exception as e:
        print(f"   ❌ GHZ state test failed: {e}")
        return False
    
    return True


def demonstrate_noise_sweep():
    """Demonstrate noise sweep testing."""
    print("\n🌊 Noise Sweep Demo")
    print("=" * 50)
    
    if not QISKIT_AVAILABLE:
        print("⚠️  Skipping noise sweep - Qiskit not available")
        return
    
    test_runner = NoiseAwareTest(default_shots=500)
    bell_circuit = create_bell_circuit()
    
    try:
        print("Running Bell state with different noise levels...")
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        results = test_runner.run_with_noise_sweep(
            bell_circuit,
            noise_levels,
            shots=500
        )
        
        print("\nNoise Level | Fidelity | Execution Time")
        print("-" * 45)
        
        for noise_level, result in results.items():
            fidelity = test_runner.calculate_bell_fidelity(result)
            print(f"{noise_level:10.3f} | {fidelity:8.3f} | {result.execution_time:8.3f}s")
        
        print("✅ Noise sweep completed successfully!")
        
    except Exception as e:
        print(f"❌ Noise sweep failed: {e}")


def demonstrate_database_setup():
    """Demonstrate database setup and migrations."""
    print("\n🗄️  Database Setup Demo")
    print("=" * 50)
    
    try:
        print("Running database migrations...")
        success = run_migrations()
        
        if success:
            print("✅ Database setup completed successfully!")
            
            # Test database connection
            from quantum_devops_ci.database.connection import get_connection
            conn = get_connection()
            
            # Test query
            result = conn.execute_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row['name'] for row in result]
            
            print(f"📊 Created {len(tables)} tables:")
            for table in tables:
                print(f"   - {table}")
                
        else:
            print("❌ Database setup failed")
            
    except Exception as e:
        print(f"❌ Database setup error: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 Quantum DevOps CI Testing Framework Demo")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    success = True
    
    # Run database setup first
    demonstrate_database_setup()
    
    # Run basic testing demo
    if not demonstrate_basic_testing():
        success = False
    
    # Run noise sweep demo
    demonstrate_noise_sweep()
    
    if success:
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Explore the quantum.config.yml configuration file")
        print("2. Set up quantum provider credentials")
        print("3. Run: quantum-test init")
        print("4. Run: quantum-test run --framework qiskit")
    else:
        print("\n⚠️  Some demonstrations had issues")
        print("Check the output above for specific error details")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())