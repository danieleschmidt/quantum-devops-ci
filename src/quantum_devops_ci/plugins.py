"""
Quantum framework adapter plugins for unified interface.
Generation 1 implementation for basic framework support.
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class QuantumFrameworkAdapter(ABC):
    """Abstract base class for quantum framework adapters."""
    
    @abstractmethod
    def create_circuit(self, num_qubits: int, num_clbits: int = 0) -> Any:
        """Create a quantum circuit with specified qubits and classical bits."""
        pass
    
    @abstractmethod
    def add_gate(self, circuit: Any, gate_name: str, qubits: List[int], **kwargs) -> None:
        """Add a quantum gate to the circuit."""
        pass
    
    @abstractmethod
    def execute_circuit(self, circuit: Any, shots: int = 1000) -> Any:
        """Execute the circuit and return results."""
        pass
    
    @abstractmethod
    def get_counts(self, result: Any) -> Dict[str, int]:
        """Extract measurement counts from result."""
        pass

class QiskitAdapter(QuantumFrameworkAdapter):
    """Adapter for Qiskit quantum framework."""
    
    def __init__(self):
        try:
            import qiskit
            from qiskit import Aer
            self.qiskit = qiskit
            self.backend = Aer.get_backend('qasm_simulator')
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("Qiskit not available")
    
    def create_circuit(self, num_qubits: int, num_clbits: int = 0) -> Any:
        """Create a Qiskit quantum circuit."""
        if not self.available:
            raise RuntimeError("Qiskit not available")
        
        from qiskit import QuantumCircuit
        return QuantumCircuit(num_qubits, num_clbits or num_qubits)
    
    def add_gate(self, circuit: Any, gate_name: str, qubits: List[int], **kwargs) -> None:
        """Add a gate to Qiskit circuit."""
        if gate_name.lower() == 'h':
            circuit.h(qubits[0])
        elif gate_name.lower() == 'x':
            circuit.x(qubits[0])
        elif gate_name.lower() == 'cx':
            circuit.cx(qubits[0], qubits[1])
        elif gate_name.lower() == 'measure':
            circuit.measure_all()
    
    def execute_circuit(self, circuit: Any, shots: int = 1000) -> Any:
        """Execute circuit on Qiskit backend."""
        if not self.available:
            raise RuntimeError("Qiskit not available")
        
        from qiskit import execute
        job = execute(circuit, self.backend, shots=shots)
        return job.result()
    
    def get_counts(self, result: Any) -> Dict[str, int]:
        """Get counts from Qiskit result."""
        return result.get_counts()

class CirqAdapter(QuantumFrameworkAdapter):
    """Adapter for Cirq quantum framework."""
    
    def __init__(self):
        try:
            import cirq
            self.cirq = cirq
            self.simulator = cirq.Simulator()
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("Cirq not available")
    
    def create_circuit(self, num_qubits: int, num_clbits: int = 0) -> Any:
        """Create a Cirq circuit."""
        if not self.available:
            raise RuntimeError("Cirq not available")
        
        import cirq
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        circuit.qubits = qubits
        return circuit
    
    def add_gate(self, circuit: Any, gate_name: str, qubits: List[int], **kwargs) -> None:
        """Add a gate to Cirq circuit."""
        import cirq
        
        circuit_qubits = circuit.qubits
        
        if gate_name.lower() == 'h':
            circuit.append(cirq.H(circuit_qubits[qubits[0]]))
        elif gate_name.lower() == 'x':
            circuit.append(cirq.X(circuit_qubits[qubits[0]]))
        elif gate_name.lower() == 'cx':
            circuit.append(cirq.CNOT(circuit_qubits[qubits[0]], circuit_qubits[qubits[1]]))
        elif gate_name.lower() == 'measure':
            circuit.append(cirq.measure(*circuit_qubits, key='result'))
    
    def execute_circuit(self, circuit: Any, shots: int = 1000) -> Any:
        """Execute circuit on Cirq simulator."""
        if not self.available:
            raise RuntimeError("Cirq not available")
        
        result = self.simulator.run(circuit, repetitions=shots)
        return result
    
    def get_counts(self, result: Any) -> Dict[str, int]:
        """Get counts from Cirq result."""
        measurements = result.measurements['result']
        counts = {}
        
        for measurement in measurements:
            bit_string = ''.join(str(bit) for bit in measurement)
            counts[bit_string] = counts.get(bit_string, 0) + 1
        
        return counts

class MockAdapter(QuantumFrameworkAdapter):
    """Mock adapter for testing when no quantum frameworks are available."""
    
    def __init__(self):
        self.available = True
    
    def create_circuit(self, num_qubits: int, num_clbits: int = 0) -> Any:
        """Create a mock circuit."""
        return {
            'num_qubits': num_qubits,
            'num_clbits': num_clbits or num_qubits,
            'gates': []
        }
    
    def add_gate(self, circuit: Any, gate_name: str, qubits: List[int], **kwargs) -> None:
        """Add a gate to mock circuit."""
        circuit['gates'].append({
            'name': gate_name,
            'qubits': qubits,
            'params': kwargs
        })
    
    def execute_circuit(self, circuit: Any, shots: int = 1000) -> Any:
        """Execute mock circuit."""
        # Return mock Bell state results
        return {
            'counts': {'00': shots // 2, '11': shots // 2},
            'shots': shots
        }
    
    def get_counts(self, result: Any) -> Dict[str, int]:
        """Get counts from mock result."""
        return result['counts']

class FrameworkRegistry:
    """Registry for quantum framework adapters."""
    
    def __init__(self):
        self.adapters = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize available framework adapters."""
        # Try to register Qiskit
        qiskit_adapter = QiskitAdapter()
        if qiskit_adapter.available:
            self.adapters['qiskit'] = qiskit_adapter
        
        # Try to register Cirq
        cirq_adapter = CirqAdapter()
        if cirq_adapter.available:
            self.adapters['cirq'] = cirq_adapter
        
        # Always register mock adapter
        self.adapters['mock'] = MockAdapter()
    
    def get_adapter(self, framework: str) -> QuantumFrameworkAdapter:
        """Get adapter for specified framework."""
        if framework in self.adapters:
            return self.adapters[framework]
        
        # Fallback to mock adapter
        logger.warning(f"Framework {framework} not available, using mock adapter")
        return self.adapters['mock']
    
    def list_available_frameworks(self) -> List[str]:
        """List all available framework adapters."""
        return list(self.adapters.keys())

# Global registry instance
framework_registry = FrameworkRegistry()

def get_framework_adapter(framework: str) -> QuantumFrameworkAdapter:
    """Get framework adapter instance."""
    return framework_registry.get_adapter(framework)

def list_available_frameworks() -> List[str]:
    """List available quantum frameworks."""
    return framework_registry.list_available_frameworks()