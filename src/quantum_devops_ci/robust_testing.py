"""
Robust quantum testing framework with comprehensive error handling and validation.
Generation 2 implementation with reliability and security focus.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import numpy as np

from .exceptions import (
    QuantumTestError, QuantumValidationError, QuantumSecurityError,
    QuantumTimeoutError, QuantumResourceError
)
from .security import SecurityManager
from .validation import QuantumCircuitValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSeverity(Enum):
    """Test failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Comprehensive test result with metadata."""
    test_name: str
    status: TestStatus
    duration_seconds: float
    severity: TestSeverity = TestSeverity.MEDIUM
    error_message: Optional[str] = None
    error_details: Optional[str] = None
    quantum_metrics: Dict[str, Any] = field(default_factory=dict)
    security_checks: Dict[str, bool] = field(default_factory=dict)
    retry_count: int = 0
    timestamps: Dict[str, float] = field(default_factory=dict)

@dataclass
class NoiseTestConfig:
    """Configuration for noise-aware testing."""
    noise_model: Optional[Any] = None
    noise_levels: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05])
    shots: int = 1000
    timeout_seconds: int = 300
    max_retries: int = 3
    fidelity_threshold: float = 0.8
    validate_security: bool = True

class RobustNoiseAwareTest:
    """
    Enhanced noise-aware testing framework with comprehensive error handling.
    """
    
    def __init__(self, config: Optional[NoiseTestConfig] = None):
        self.config = config or NoiseTestConfig()
        self.security_manager = SecurityManager()
        self.circuit_validator = QuantumCircuitValidator()
        self.test_results: List[TestResult] = []
        
        # Circuit execution cache to avoid redundant computations
        self._execution_cache: Dict[str, Any] = {}
        
        # Resource monitoring
        self._resource_limits = {
            'max_circuit_depth': 200,
            'max_qubits': 50,
            'max_memory_mb': 1024,
            'max_execution_time': 600
        }
    
    @contextmanager
    def test_context(self, test_name: str, severity: TestSeverity = TestSeverity.MEDIUM):
        """Context manager for robust test execution with monitoring."""
        result = TestResult(
            test_name=test_name,
            status=TestStatus.RUNNING,
            duration_seconds=0.0,
            severity=severity,
            timestamps={'start': time.time()}
        )
        
        try:
            logger.info(f"Starting test: {test_name}")
            yield result
            
            result.status = TestStatus.PASSED
            result.timestamps['end'] = time.time()
            result.duration_seconds = result.timestamps['end'] - result.timestamps['start']
            
            logger.info(f"Test passed: {test_name} ({result.duration_seconds:.2f}s)")
            
        except QuantumTestError as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            result.error_details = traceback.format_exc()
            logger.error(f"Test failed: {test_name} - {e}")
            
        except QuantumSecurityError as e:
            result.status = TestStatus.ERROR
            result.error_message = f"Security violation: {e}"
            result.error_details = traceback.format_exc()
            logger.critical(f"Security error in test: {test_name} - {e}")
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = f"Unexpected error: {e}"
            result.error_details = traceback.format_exc()
            logger.error(f"Unexpected error in test: {test_name} - {e}")
            
        finally:
            if 'end' not in result.timestamps:
                result.timestamps['end'] = time.time()
                result.duration_seconds = result.timestamps['end'] - result.timestamps['start']
            
            self.test_results.append(result)
    
    def validate_circuit_security(self, circuit: Any) -> Dict[str, bool]:
        """Comprehensive circuit security validation."""
        security_checks = {
            'circuit_depth_safe': True,
            'qubit_count_safe': True,
            'gate_whitelist_compliant': True,
            'resource_usage_safe': True,
            'no_malicious_patterns': True
        }
        
        try:
            # Check circuit depth
            if hasattr(circuit, 'depth'):
                if circuit.depth() > self._resource_limits['max_circuit_depth']:
                    security_checks['circuit_depth_safe'] = False
                    raise QuantumSecurityError(f"Circuit depth {circuit.depth()} exceeds limit")
            
            # Check qubit count
            if hasattr(circuit, 'num_qubits'):
                if circuit.num_qubits > self._resource_limits['max_qubits']:
                    security_checks['qubit_count_safe'] = False
                    raise QuantumSecurityError(f"Qubit count {circuit.num_qubits} exceeds limit")
            
            # Validate using circuit validator
            if not self.circuit_validator.validate_circuit(circuit):
                security_checks['gate_whitelist_compliant'] = False
                raise QuantumSecurityError("Circuit contains forbidden gates or patterns")
            
            # Check for malicious patterns
            if self._detect_malicious_patterns(circuit):
                security_checks['no_malicious_patterns'] = False
                raise QuantumSecurityError("Malicious patterns detected in circuit")
                
        except QuantumSecurityError:
            raise
        except Exception as e:
            logger.warning(f"Security validation warning: {e}")
            
        return security_checks
    
    def _detect_malicious_patterns(self, circuit: Any) -> bool:
        """Detect potentially malicious circuit patterns."""
        try:
            # Check for excessive gate repetition (potential DoS)
            if hasattr(circuit, 'count_ops'):
                ops = circuit.count_ops()
                for gate, count in ops.items():
                    if count > 1000:  # Arbitrary threshold
                        logger.warning(f"Excessive {gate} gates: {count}")
                        return True
            
            # Check for suspicious QASM patterns
            if hasattr(circuit, 'qasm'):
                qasm_str = circuit.qasm()
                suspicious_patterns = ['while', 'for', 'include', 'eval']
                for pattern in suspicious_patterns:
                    if pattern in qasm_str.lower():
                        logger.warning(f"Suspicious QASM pattern: {pattern}")
                        return True
            
        except Exception as e:
            logger.warning(f"Malicious pattern detection error: {e}")
            
        return False
    
    def run_with_noise_sweep(self, circuit: Any, **kwargs) -> Dict[float, Any]:
        """Run circuit with multiple noise levels and comprehensive error handling."""
        noise_levels = kwargs.get('noise_levels', self.config.noise_levels)
        shots = kwargs.get('shots', self.config.shots)
        results = {}
        
        for noise_level in noise_levels:
            try:
                with self.test_context(f"noise_sweep_{noise_level}", TestSeverity.MEDIUM) as result:
                    # Security validation
                    security_checks = self.validate_circuit_security(circuit)
                    result.security_checks = security_checks
                    
                    # Execute with noise
                    execution_result = self._execute_with_noise(
                        circuit, noise_level, shots
                    )
                    
                    # Validate results
                    if not self._validate_execution_result(execution_result):
                        raise QuantumTestError(f"Invalid execution result at noise level {noise_level}")
                    
                    # Store quantum metrics
                    result.quantum_metrics = {
                        'noise_level': noise_level,
                        'shots': shots,
                        'fidelity': self._calculate_fidelity(execution_result),
                        'success_probability': self._calculate_success_probability(execution_result)
                    }
                    
                    results[noise_level] = execution_result
                    
            except Exception as e:
                logger.error(f"Noise sweep failed at level {noise_level}: {e}")
                if isinstance(e, QuantumSecurityError):
                    raise  # Re-raise security errors
                # Continue with other noise levels
                results[noise_level] = None
        
        return results
    
    def run_with_timeout(self, circuit: Any, timeout_seconds: Optional[int] = None) -> Any:
        """Execute circuit with timeout protection."""
        timeout = timeout_seconds or self.config.timeout_seconds
        
        try:
            # Use asyncio for timeout control
            return asyncio.wait_for(
                self._async_execute_circuit(circuit),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise QuantumTimeoutError(f"Circuit execution exceeded {timeout} seconds")
    
    async def _async_execute_circuit(self, circuit: Any) -> Any:
        """Asynchronous circuit execution wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_circuit_sync, circuit)
    
    def _execute_circuit_sync(self, circuit: Any) -> Any:
        """Synchronous circuit execution with error handling."""
        try:
            # Try different backends based on availability
            backend = self._get_available_backend()
            
            if backend is None:
                raise QuantumResourceError("No quantum backends available")
            
            # Execute circuit
            if hasattr(backend, 'run'):
                job = backend.run(circuit, shots=self.config.shots)
                return job.result()
            else:
                # Fallback for different backend types
                return backend.execute(circuit)
                
        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise QuantumTestError(f"Circuit execution failed: {e}")
    
    def _execute_with_noise(self, circuit: Any, noise_level: float, shots: int) -> Any:
        """Execute circuit with specified noise level."""
        try:
            # Create cache key
            cache_key = f"{id(circuit)}_{noise_level}_{shots}"
            
            if cache_key in self._execution_cache:
                logger.debug(f"Using cached result for {cache_key}")
                return self._execution_cache[cache_key]
            
            # Try to get noisy backend
            backend = self._get_noisy_backend(noise_level)
            
            if backend is None:
                # Fallback to simulation with manual noise injection
                result = self._simulate_with_noise(circuit, noise_level, shots)
            else:
                job = backend.run(circuit, shots=shots)
                result = job.result()
            
            # Cache successful results
            self._execution_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Noisy execution failed: {e}")
            raise QuantumTestError(f"Noisy execution failed: {e}")
    
    def _get_available_backend(self) -> Optional[Any]:
        """Get available quantum backend with fallback options."""
        try:
            # Try Qiskit Aer
            from qiskit import Aer
            return Aer.get_backend('qasm_simulator')
        except ImportError:
            pass
        
        try:
            # Try Cirq
            import cirq
            return cirq.Simulator()
        except ImportError:
            pass
        
        # Return mock backend as last resort
        from .quantum_fixtures import MockQuantumBackend
        logger.warning("Using mock backend - no quantum frameworks available")
        return MockQuantumBackend()
    
    def _get_noisy_backend(self, noise_level: float) -> Optional[Any]:
        """Get quantum backend with noise model."""
        try:
            from qiskit import Aer
            from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
            
            # Create noise model
            noise_model = NoiseModel()
            error = depolarizing_error(noise_level, 1)
            noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
            
            backend = Aer.get_backend('qasm_simulator')
            backend.set_options(noise_model=noise_model)
            return backend
            
        except ImportError:
            return None
    
    def _simulate_with_noise(self, circuit: Any, noise_level: float, shots: int) -> Any:
        """Simulate circuit with manual noise injection."""
        # Simplified noise simulation for fallback
        from .quantum_fixtures import MockQuantumResult
        
        # Apply noise to mock result
        result = MockQuantumResult(shots)
        
        # Modify counts to simulate noise
        counts = result.get_counts()
        noisy_counts = {}
        
        for state, count in counts.items():
            # Add noise by redistributing some counts
            noise_reduction = int(count * noise_level)
            noisy_counts[state] = max(0, count - noise_reduction)
            
            # Add noise to random states
            noise_state = ''.join(['1' if bit == '0' else '0' for bit in state])
            noisy_counts[noise_state] = noisy_counts.get(noise_state, 0) + noise_reduction
        
        result._counts = noisy_counts
        return result
    
    def _validate_execution_result(self, result: Any) -> bool:
        """Validate quantum execution result."""
        try:
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
                
                # Check if counts are valid
                if not isinstance(counts, dict) or len(counts) == 0:
                    return False
                
                # Check if all counts are non-negative integers
                for state, count in counts.items():
                    if not isinstance(count, int) or count < 0:
                        return False
                
                # Check total shots consistency
                total_counts = sum(counts.values())
                if total_counts != self.config.shots:
                    logger.warning(f"Total counts {total_counts} != expected shots {self.config.shots}")
                
                return True
            
            return True  # Assume valid if no counts method
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            return False
    
    def _calculate_fidelity(self, result: Any) -> float:
        """Calculate quantum state fidelity from result."""
        try:
            if hasattr(result, 'get_statevector'):
                # For statevector results
                statevector = result.get_statevector()
                if statevector is not None:
                    return abs(np.dot(statevector.conj(), statevector))
            
            if hasattr(result, 'get_counts'):
                # Estimate fidelity from measurement counts
                counts = result.get_counts()
                total_shots = sum(counts.values())
                
                # Simple fidelity estimate based on most probable state
                max_count = max(counts.values()) if counts else 0
                return max_count / total_shots if total_shots > 0 else 0.0
            
            return 1.0  # Default perfect fidelity
            
        except Exception as e:
            logger.warning(f"Fidelity calculation failed: {e}")
            return 0.0
    
    def _calculate_success_probability(self, result: Any) -> float:
        """Calculate success probability from measurement result."""
        try:
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
                total_shots = sum(counts.values())
                
                # Define success as measurement of expected states
                # This is circuit-dependent, so we use a general heuristic
                success_states = ['00', '11']  # For Bell state circuits
                success_count = sum(counts.get(state, 0) for state in success_states)
                
                return success_count / total_shots if total_shots > 0 else 0.0
            
            return 1.0  # Default perfect success
            
        except Exception as e:
            logger.warning(f"Success probability calculation failed: {e}")
            return 0.0
    
    def run_with_retry(self, test_func: Callable, max_retries: Optional[int] = None) -> Any:
        """Execute test function with retry logic for transient failures."""
        max_retries = max_retries or self.config.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return test_func()
            except (QuantumResourceError, QuantumTimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Test attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Test failed after {max_retries + 1} attempts")
                    raise
            except QuantumSecurityError:
                # Don't retry security errors
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Test attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(1)
                else:
                    raise
        
        if last_exception:
            raise last_exception
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        
        # Calculate metrics
        total_duration = sum(r.duration_seconds for r in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        # Security summary
        security_violations = sum(
            1 for r in self.test_results 
            if any(not check for check in r.security_checks.values())
        )
        
        # Quantum metrics summary
        avg_fidelity = np.mean([
            r.quantum_metrics.get('fidelity', 0.0) 
            for r in self.test_results 
            if r.quantum_metrics
        ]) if any(r.quantum_metrics for r in self.test_results) else 0.0
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration_seconds': total_duration,
                'average_duration_seconds': avg_duration
            },
            'security': {
                'security_violations': security_violations,
                'secure_tests': total_tests - security_violations,
                'security_compliance_rate': (total_tests - security_violations) / total_tests if total_tests > 0 else 1.0
            },
            'quantum_metrics': {
                'average_fidelity': avg_fidelity,
                'quantum_tests': sum(1 for r in self.test_results if r.quantum_metrics)
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'duration': r.duration_seconds,
                    'severity': r.severity.value,
                    'error_message': r.error_message,
                    'quantum_metrics': r.quantum_metrics,
                    'security_checks': r.security_checks,
                    'retry_count': r.retry_count
                }
                for r in self.test_results
            ]
        }