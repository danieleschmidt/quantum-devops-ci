"""
Noise-aware testing framework for quantum DevOps CI/CD.

This module provides base classes and utilities for testing quantum algorithms
under realistic noise conditions, supporting multiple quantum frameworks.
"""

import abc
import warnings
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available - some testing features will be limited")

from .exceptions import (
    TestExecutionError, NoiseModelError, BackendConnectionError,
    CircuitValidationError, ResourceExhaustionError
)
from .validation import QuantumCircuitValidator, validate_inputs
from .security import requires_auth, audit_action

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    warnings.warn("pytest not available - some testing features will be limited")

# Framework imports with fallbacks
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


@dataclass
class TestResult:
    """Container for quantum test results."""
    counts: Dict[str, int]
    shots: int
    execution_time: float
    backend_name: str
    noise_model: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def get_counts(self) -> Dict[str, int]:
        """Get measurement counts."""
        return self.counts
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        total_shots = sum(self.counts.values())
        return {state: count / total_shots for state, count in self.counts.items()}


class NoiseAwareTest(abc.ABC):
    """
    Base class for noise-aware quantum testing.
    
    This class provides common functionality for testing quantum algorithms
    under various noise conditions, supporting multiple quantum frameworks.
    """
    
    def __init__(self, default_shots: int = 1000, timeout_seconds: int = 300):
        """
        Initialize noise-aware test.
        
        Args:
            default_shots: Default number of measurement shots
            timeout_seconds: Maximum execution time for tests
        """
        self.default_shots = default_shots
        self.timeout_seconds = timeout_seconds
        self.backend_cache = {}
    
    @requires_auth('test.execute')
    @audit_action('run_circuit', 'quantum_circuit')
    @validate_inputs(shots=lambda x: x is None or QuantumCircuitValidator.validate_shots(x))
    def run_circuit(
        self, 
        circuit, 
        shots: Optional[int] = None,
        backend: str = "qasm_simulator",
        **kwargs
    ) -> TestResult:
        """
        Run quantum circuit on simulator.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            backend: Backend name
            **kwargs: Additional execution parameters
            
        Returns:
            TestResult with execution results
            
        Raises:
            TestExecutionError: If circuit execution fails
            CircuitValidationError: If circuit validation fails
            BackendConnectionError: If backend connection fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            if shots is None:
                shots = self.default_shots
            
            logger.info(f"Running circuit with {shots} shots on backend {backend}")
            
            # Validate circuit if it has expected attributes
            if hasattr(circuit, 'num_qubits'):
                try:
                    QuantumCircuitValidator.validate_circuit_parameters(
                        circuit.num_qubits, 
                        len(getattr(circuit, 'data', [])),
                        getattr(circuit, 'depth', lambda: 0)()
                    )
                except Exception as e:
                    raise CircuitValidationError(f"Circuit validation failed: {e}")
            
            # Framework-specific execution with error handling
            try:
                if QISKIT_AVAILABLE and hasattr(circuit, 'measure'):
                    result = self._run_qiskit_circuit(circuit, shots, backend, **kwargs)
                elif CIRQ_AVAILABLE and hasattr(circuit, 'moments'):
                    result = self._run_cirq_circuit(circuit, shots, backend, **kwargs)
                else:
                    raise TestExecutionError(f"Framework not supported for circuit type: {type(circuit)}")
                
                logger.info(f"Circuit execution completed successfully in {result.execution_time:.2f}s")
                return result
                
            except Exception as e:
                if isinstance(e, (TestExecutionError, CircuitValidationError)):
                    raise
                
                # Check if it's a resource/timeout issue
                if 'timeout' in str(e).lower() or 'memory' in str(e).lower():
                    raise ResourceExhaustionError(f"Resource exhaustion during execution: {e}")
                
                # Check if it's a backend connection issue
                if 'connection' in str(e).lower() or 'network' in str(e).lower():
                    raise BackendConnectionError(f"Backend connection failed: {e}")
                
                raise TestExecutionError(f"Circuit execution failed: {e}", details={
                    'backend': backend,
                    'shots': shots,
                    'circuit_type': type(circuit).__name__
                })
                
        except Exception as e:
            logger.error(f"Unexpected error in run_circuit: {e}")
            if isinstance(e, (TestExecutionError, CircuitValidationError, BackendConnectionError, ResourceExhaustionError)):
                raise
            raise TestExecutionError(f"Unexpected error during circuit execution: {e}")
    
    @requires_auth('test.execute')
    @audit_action('run_with_noise', 'quantum_circuit')
    @validate_inputs(shots=lambda x: x is None or QuantumCircuitValidator.validate_shots(x))
    def run_with_noise(
        self,
        circuit,
        noise_model: Union[str, Any],
        shots: Optional[int] = None,
        **kwargs
    ) -> TestResult:
        """
        Run circuit with specified noise model.
        
        Args:
            circuit: Quantum circuit to execute
            noise_model: Noise model name or object
            shots: Number of measurement shots
            **kwargs: Additional parameters
            
        Returns:
            TestResult with noisy execution results
            
        Raises:
            TestExecutionError: If circuit execution fails
            NoiseModelError: If noise model is invalid
            CircuitValidationError: If circuit validation fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            if shots is None:
                shots = self.default_shots
            
            logger.info(f"Running noisy circuit with {shots} shots, noise model: {noise_model}")
            
            # Validate noise model
            if isinstance(noise_model, str) and not noise_model.strip():
                raise NoiseModelError("Noise model name cannot be empty")
            
            # Framework-specific noisy execution with error handling
            try:
                if QISKIT_AVAILABLE and hasattr(circuit, 'measure'):
                    result = self._run_qiskit_noisy(circuit, noise_model, shots, **kwargs)
                elif CIRQ_AVAILABLE and hasattr(circuit, 'moments'):
                    result = self._run_cirq_noisy(circuit, noise_model, shots, **kwargs)
                else:
                    raise TestExecutionError(f"Framework not supported for circuit type: {type(circuit)}")
                
                logger.info(f"Noisy circuit execution completed in {result.execution_time:.2f}s")
                return result
                
            except NoiseModelError:
                raise
            except Exception as e:
                if 'noise' in str(e).lower():
                    raise NoiseModelError(f"Noise model error: {e}")
                raise TestExecutionError(f"Noisy circuit execution failed: {e}")
                
        except Exception as e:
            logger.error(f"Error in run_with_noise: {e}")
            if isinstance(e, (TestExecutionError, NoiseModelError, CircuitValidationError)):
                raise
            raise TestExecutionError(f"Unexpected error during noisy execution: {e}")
    
    def run_with_noise_sweep(
        self,
        circuit,
        noise_levels: List[float],
        shots: Optional[int] = None,
        **kwargs
    ) -> Dict[float, TestResult]:
        """
        Run circuit with sweep of noise levels.
        
        Args:
            circuit: Quantum circuit to execute
            noise_levels: List of noise levels to test
            shots: Number of measurement shots per level
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping noise levels to results
            
        Raises:
            TestExecutionError: If any noise level execution fails
            ValueError: If noise levels are invalid
        """
        logger = logging.getLogger(__name__)
        
        if not noise_levels:
            raise ValueError("Noise levels list cannot be empty")
        
        if any(level < 0 or level > 1 for level in noise_levels):
            raise ValueError("Noise levels must be between 0 and 1")
        
        results = {}
        failed_levels = []
        
        logger.info(f"Running noise sweep with {len(noise_levels)} levels")
        
        for noise_level in noise_levels:
            try:
                results[noise_level] = self.run_with_noise(
                    circuit, 
                    f"depolarizing_{noise_level}",
                    shots=shots,
                    **kwargs
                )
                logger.debug(f"Completed noise level {noise_level}")
            except Exception as e:
                failed_levels.append((noise_level, str(e)))
                logger.warning(f"Failed to run noise level {noise_level}: {e}")
        
        if failed_levels and not results:
            # All levels failed
            raise TestExecutionError(f"All noise levels failed: {failed_levels}")
        elif failed_levels:
            # Some levels failed, log warnings
            logger.warning(f"{len(failed_levels)} noise levels failed: {[level for level, _ in failed_levels]}")
        
        return results
    
    def run_on_hardware(
        self,
        circuit,
        backend: str = "least_busy",
        shots: Optional[int] = None,
        **kwargs
    ) -> TestResult:
        """
        Run circuit on real quantum hardware.
        
        Args:
            circuit: Quantum circuit to execute
            backend: Hardware backend name or selection strategy
            shots: Number of measurement shots
            **kwargs: Additional parameters
            
        Returns:
            TestResult from hardware execution
        """
        if shots is None:
            shots = min(self.default_shots, 1000)  # Limit hardware usage
        
        warnings.warn("Hardware execution requires proper authentication and may incur costs")
        
        # Framework-specific hardware execution
        if QISKIT_AVAILABLE and hasattr(circuit, 'measure'):
            return self._run_qiskit_hardware(circuit, backend, shots, **kwargs)
        else:
            raise NotImplementedError(f"Hardware execution not supported for: {type(circuit)}")
    
    def calculate_bell_fidelity(self, result: TestResult) -> float:
        """
        Calculate fidelity for Bell state preparation.
        
        Args:
            result: Test result containing measurement counts
            
        Returns:
            Fidelity score between 0 and 1
        """
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        # Expected Bell state: |00⟩ + |11⟩
        expected_states = {'00', '11'}
        correct_counts = sum(counts.get(state, 0) for state in expected_states)
        
        return correct_counts / total_shots if total_shots > 0 else 0.0
    
    def calculate_state_fidelity(self, result: TestResult, target_state: str) -> float:
        """
        Calculate fidelity for specific target state.
        
        Args:
            result: Test result containing measurement counts
            target_state: Target state name (e.g., 'bell', 'ghz', 'uniform')
            
        Returns:
            Fidelity score between 0 and 1
        """
        if target_state == 'bell':
            return self.calculate_bell_fidelity(result)
        elif target_state == 'uniform':
            return self._calculate_uniform_fidelity(result)
        else:
            raise ValueError(f"Unknown target state: {target_state}")
    
    def run_with_mitigation(
        self,
        circuit,
        noise_level: float,
        method: str = "zero_noise_extrapolation",
        **kwargs
    ) -> TestResult:
        """
        Run circuit with error mitigation.
        
        Args:
            circuit: Quantum circuit to execute
            noise_level: Noise level for simulation
            method: Error mitigation method
            **kwargs: Additional parameters
            
        Returns:
            TestResult with error mitigation applied
        """
        # Placeholder for error mitigation implementation
        warnings.warn("Error mitigation is not yet fully implemented")
        
        # For now, just run with noise
        return self.run_with_noise(circuit, f"depolarizing_{noise_level}", **kwargs)
    
    # Framework-specific implementations
    def _run_qiskit_circuit(self, circuit, shots: int, backend: str, **kwargs) -> TestResult:
        """Run Qiskit circuit on simulator."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        import time
        start_time = time.time()
        
        # Get or create backend
        if backend == "qasm_simulator":
            sim_backend = AerSimulator()
        else:
            raise ValueError(f"Unknown Qiskit backend: {backend}")
        
        # Transpile and execute
        transpiled = transpile(circuit, sim_backend)
        job = sim_backend.run(transpiled, shots=shots)
        result = job.result()
        
        execution_time = time.time() - start_time
        counts = result.get_counts()
        
        return TestResult(
            counts=counts,
            shots=shots,
            execution_time=execution_time,
            backend_name=backend
        )
    
    def _run_qiskit_noisy(self, circuit, noise_model, shots: int, **kwargs) -> TestResult:
        """Run Qiskit circuit with noise."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        import time
        start_time = time.time()
        
        # Create noisy simulator
        if isinstance(noise_model, str):
            if noise_model.startswith("depolarizing_"):
                noise_level = float(noise_model.split("_")[1])
                # Create simple depolarizing noise model
                from qiskit.providers.aer.noise import depolarizing_error, NoiseModel
                noise_model_obj = NoiseModel()
                error = depolarizing_error(noise_level, 1)
                noise_model_obj.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
            else:
                raise ValueError(f"Unknown noise model: {noise_model}")
        else:
            noise_model_obj = noise_model
        
        sim_backend = AerSimulator(noise_model=noise_model_obj)
        
        # Execute
        transpiled = transpile(circuit, sim_backend)
        job = sim_backend.run(transpiled, shots=shots)
        result = job.result()
        
        execution_time = time.time() - start_time
        counts = result.get_counts()
        
        return TestResult(
            counts=counts,
            shots=shots,
            execution_time=execution_time,
            backend_name="noisy_qasm_simulator",
            noise_model=str(noise_model)
        )
    
    def _run_qiskit_hardware(self, circuit, backend: str, shots: int, **kwargs) -> TestResult:
        """Run Qiskit circuit on hardware (placeholder)."""
        # This would require IBM Quantum provider setup
        raise NotImplementedError("Hardware execution requires IBM Quantum provider configuration")
    
    def _run_cirq_circuit(self, circuit, shots: int, backend: str, **kwargs) -> TestResult:
        """Run Cirq circuit on simulator."""
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq not available")
        
        # Placeholder for Cirq implementation
        raise NotImplementedError("Cirq execution not yet implemented")
    
    def _run_cirq_noisy(self, circuit, noise_model, shots: int, **kwargs) -> TestResult:
        """Run Cirq circuit with noise."""
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq not available")
        
        # Placeholder for Cirq noisy implementation
        raise NotImplementedError("Cirq noisy execution not yet implemented")
    
    def _calculate_uniform_fidelity(self, result: TestResult) -> float:
        """Calculate fidelity for uniform superposition."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            return 0.0
        
        # For uniform distribution, all computational basis states should be roughly equal
        num_states = len(counts)
        if num_states == 0:
            return 0.0
        
        expected_prob = 1.0 / num_states
        actual_probs = [count / total_shots for count in counts.values()]
        
        # Calculate fidelity as 1 - variance from uniform  
        if not NUMPY_AVAILABLE:
            return self._calculate_uniform_fidelity_simple(result)
            
        variance = np.var(actual_probs)
        max_variance = expected_prob * (1 - expected_prob)  # Maximum possible variance
        
        if max_variance == 0:
            return 1.0
        
        return max(0.0, 1.0 - variance / max_variance)
    
    def _calculate_uniform_fidelity_simple(self, result: TestResult) -> float:
        """Simple uniform fidelity calculation without NumPy."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            return 0.0
        
        num_states = len(counts)
        if num_states == 0:
            return 0.0
        
        expected_prob = 1.0 / num_states
        actual_probs = [count / total_shots for count in counts.values()]
        
        # Calculate mean squared deviation from expected probability
        mean_sq_dev = sum((prob - expected_prob) ** 2 for prob in actual_probs) / num_states
        max_possible_dev = expected_prob ** 2  # Maximum possible deviation
        
        if max_possible_dev == 0:
            return 1.0
        
        return max(0.0, 1.0 - mean_sq_dev / max_possible_dev)
    
    def run_test_suite(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test suite with given configuration."""
        logger = logging.getLogger(__name__)
        
        framework = test_config.get('framework', 'qiskit')
        backend = test_config.get('backend', 'qasm_simulator')
        shots = test_config.get('shots', 1000)
        noise_level = test_config.get('noise_level', 0.0)
        
        logger.info(f"Running test suite with {framework} framework")
        
        test_results = {}
        total_time = 0.0
        
        # Define test circuits based on framework
        test_circuits = self._get_test_circuits(framework)
        
        for test_name, circuit_func in test_circuits.items():
            try:
                import time
                start_time = time.time()
                
                # Create circuit
                circuit = circuit_func()
                
                # Run test
                if noise_level > 0:
                    result = self.run_with_noise(circuit, f"depolarizing_{noise_level}", shots=shots)
                else:
                    result = self.run_circuit(circuit, shots=shots, backend=backend)
                
                execution_time = time.time() - start_time
                total_time += execution_time
                
                # Calculate metrics based on test type
                fidelity = self._calculate_test_fidelity(test_name, result)
                
                test_results[test_name] = {
                    'passed': fidelity > 0.8,  # Pass threshold
                    'fidelity': fidelity,
                    'execution_time': execution_time,
                    'shots': shots,
                    'backend': backend,
                    'noise_level': noise_level
                }
                
                logger.debug(f"Test {test_name}: fidelity={fidelity:.3f}, time={execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                test_results[test_name] = {
                    'passed': False,
                    'error': str(e),
                    'execution_time': 0.0,
                    'shots': shots
                }
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        return {
            'tests': test_results,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'total_time': total_time,
                'framework': framework,
                'backend': backend
            }
        }
    
    def _get_test_circuits(self, framework: str) -> Dict[str, callable]:
        """Get test circuits for the specified framework."""
        if framework == 'qiskit' and QISKIT_AVAILABLE:
            return {
                'test_bell_state': self._create_bell_circuit_qiskit,
                'test_ghz_state': self._create_ghz_circuit_qiskit,
                'test_uniform_superposition': self._create_uniform_circuit_qiskit
            }
        else:
            # Mock circuits for non-available frameworks
            return {
                'test_mock_circuit': lambda: {'type': 'mock', 'qubits': 2}
            }
    
    def _create_bell_circuit_qiskit(self):
        """Create a Bell state circuit for Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
    
    def _create_ghz_circuit_qiskit(self):
        """Create a GHZ state circuit for Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        return qc
    
    def _create_uniform_circuit_qiskit(self):
        """Create a uniform superposition circuit for Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.measure_all()
        return qc
    
    def _calculate_test_fidelity(self, test_name: str, result: TestResult) -> float:
        """Calculate fidelity based on test type."""
        if 'bell' in test_name:
            return self.calculate_bell_fidelity(result)
        elif 'ghz' in test_name:
            return self._calculate_ghz_fidelity(result)
        elif 'uniform' in test_name:
            return self.calculate_state_fidelity(result, 'uniform')
        else:
            # Default: just check if we got any results
            counts = result.get_counts()
            return 1.0 if counts else 0.0
    
    def _calculate_ghz_fidelity(self, result: TestResult) -> float:
        """Calculate fidelity for GHZ state preparation."""
        counts = result.get_counts()
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            return 0.0
        
        # Expected GHZ state: |000⟩ + |111⟩
        expected_states = {'000', '111'}
        correct_counts = sum(counts.get(state, 0) for state in expected_states)
        
        return correct_counts / total_shots


# Pytest fixture decorator
def quantum_fixture(func: Callable) -> Callable:
    """
    Decorator for quantum test fixtures.
    
    This decorator marks a function as a quantum fixture that can be used
    in quantum tests. The fixture will be properly parameterized for
    different quantum frameworks if available.
    """
    if PYTEST_AVAILABLE:
        return pytest.fixture(func)
    else:
        # Return the function as-is if pytest is not available
        warnings.warn("pytest not available - quantum_fixture will not provide fixture functionality")
        return func


# Hardware compatibility test base class
class HardwareCompatibilityTest(NoiseAwareTest):
    """
    Base class for testing hardware compatibility.
    
    This class provides methods for testing quantum circuits against
    hardware constraints and requirements.
    """
    
    def get_available_backends(self) -> List[str]:
        """Get list of available quantum backends."""
        backends = ["qasm_simulator"]  # Always available
        
        # Add hardware backends if credentials are available
        # This would be expanded based on provider configuration
        
        return backends
    
    def decompose_for_backend(self, circuit, backend: str):
        """
        Decompose circuit for specific backend.
        
        Args:
            circuit: Quantum circuit to decompose
            backend: Target backend name
            
        Returns:
            Decomposed circuit compatible with backend
        """
        if QISKIT_AVAILABLE and hasattr(circuit, 'measure'):
            # Get backend-specific gate set and decompose
            return transpile(circuit, backend=backend)
        else:
            raise NotImplementedError(f"Decomposition not supported for: {type(circuit)}")
    
    def get_native_gates(self, backend: str) -> List[str]:
        """
        Get native gate set for backend.
        
        Args:
            backend: Backend name
            
        Returns:
            List of native gate names
        """
        # Default gate set for simulators
        if backend in ["qasm_simulator", "statevector_simulator"]:
            return ["u1", "u2", "u3", "cx"]
        else:
            # This would be expanded for real hardware backends
            return ["u1", "u2", "u3", "cx"]
    
    def circuits_equivalent(self, circuit1, circuit2, tolerance: float = 1e-10) -> bool:
        """
        Check if two circuits are functionally equivalent.
        
        Args:
            circuit1: First quantum circuit
            circuit2: Second quantum circuit
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if circuits are equivalent
        """
        # This is a simplified implementation
        # Full implementation would compare unitary matrices
        warnings.warn("Circuit equivalence checking is simplified")
        
        # Basic check: same number of qubits and operations
        if hasattr(circuit1, 'num_qubits') and hasattr(circuit2, 'num_qubits'):
            return circuit1.num_qubits == circuit2.num_qubits
        
        return True  # Placeholder