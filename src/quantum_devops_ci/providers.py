"""
Quantum provider integration framework.

This module provides a unified interface for interacting with quantum
computing providers like IBM Quantum, AWS Braket, Google Quantum AI,
and others through a plugin-based architecture.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .exceptions import (
    BackendConnectionError, 
    ResourceExhaustionError, 
    TestExecutionError,
    ConfigurationError
)
from .resilience import circuit_breaker, retry, timeout, CircuitBreakerConfig, RetryPolicy
from .testing import TestResult


class ProviderType(Enum):
    """Supported quantum provider types."""
    IBM_QUANTUM = "ibmq"
    AWS_BRAKET = "aws_braket" 
    GOOGLE_QUANTUM = "google_quantum"
    AZURE_QUANTUM = "azure_quantum"
    IONQ = "ionq"
    RIGETTI = "rigetti"
    SIMULATOR = "simulator"


class JobStatus(Enum):
    """Quantum job status states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProviderCredentials:
    """Provider authentication credentials."""
    provider: ProviderType
    token: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    project_id: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if credentials have required fields."""
        if self.provider == ProviderType.IBM_QUANTUM:
            return self.token is not None
        elif self.provider == ProviderType.AWS_BRAKET:
            return self.api_key is not None and self.secret_key is not None
        elif self.provider == ProviderType.GOOGLE_QUANTUM:
            return self.project_id is not None
        else:
            return True  # Simulators don't need credentials


@dataclass
class QuantumBackend:
    """Quantum backend information."""
    name: str
    provider: ProviderType
    backend_type: str  # "simulator" or "qpu"
    num_qubits: int
    max_shots: int
    cost_per_shot: float = 0.0
    cost_currency: str = "USD"
    queue_length: int = 0
    operational: bool = True
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumJob:
    """Quantum job information."""
    job_id: str
    provider: ProviderType
    backend_name: str
    status: JobStatus
    submitted_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    shots: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumProvider(ABC):
    """Abstract base class for quantum providers."""
    
    def __init__(self, credentials: ProviderCredentials):
        self.credentials = credentials
        self.provider_type = credentials.provider
        self._backends: Dict[str, QuantumBackend] = {}
        self._jobs: Dict[str, QuantumJob] = {}
        
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the provider."""
        pass
    
    @abstractmethod
    def get_backends(self, refresh: bool = False) -> List[QuantumBackend]:
        """Get list of available backends."""
        pass
    
    @abstractmethod
    def submit_job(
        self, 
        circuit: Any, 
        backend_name: str, 
        shots: int,
        **kwargs
    ) -> QuantumJob:
        """Submit quantum job for execution."""
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> QuantumJob:
        """Get status of submitted job."""
        pass
    
    @abstractmethod
    def get_job_result(self, job_id: str) -> TestResult:
        """Get results of completed job."""
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        pass
    
    def get_backend(self, name: str) -> Optional[QuantumBackend]:
        """Get specific backend by name."""
        if not self._backends:
            self.get_backends()
        return self._backends.get(name)
    
    def get_least_busy_backend(self, min_qubits: int = 1) -> Optional[QuantumBackend]:
        """Get the least busy backend with minimum qubits."""
        backends = [
            b for b in self.get_backends() 
            if b.operational and b.num_qubits >= min_qubits
        ]
        
        if not backends:
            return None
        
        return min(backends, key=lambda b: b.queue_length)
    
    def estimate_cost(self, backend_name: str, shots: int) -> float:
        """Estimate cost for running job."""
        backend = self.get_backend(backend_name)
        if not backend:
            return 0.0
        
        return backend.cost_per_shot * shots
    
    def wait_for_job(
        self, 
        job_id: str, 
        timeout_seconds: float = 3600,
        poll_interval: float = 5.0
    ) -> QuantumJob:
        """Wait for job to complete."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            job = self.get_job_status(job_id)
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job
            
            import time
            time.sleep(poll_interval)
        
        raise ResourceExhaustionError(f"Job {job_id} timed out after {timeout_seconds}s")


class IBMQuantumProvider(QuantumProvider):
    """IBM Quantum provider implementation."""
    
    def __init__(self, credentials: ProviderCredentials):
        super().__init__(credentials)
        self._service = None
        
    @circuit_breaker('ibm_auth', CircuitBreakerConfig(failure_threshold=3))
    @retry(RetryPolicy(max_attempts=3, base_delay=2.0))
    def authenticate(self) -> bool:
        """Authenticate with IBM Quantum."""
        try:
            # Try to import IBM Quantum Runtime
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            if not self.credentials.token:
                raise ConfigurationError("IBM Quantum token is required")
            
            # Initialize service
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.credentials.token
            )
            
            # Test connection by getting backends
            backends = self._service.backends()
            return len(backends) > 0
            
        except ImportError:
            warnings.warn("qiskit-ibm-runtime not available, using mock IBM provider")
            return self._mock_authenticate()
        except Exception as e:
            raise BackendConnectionError(f"Failed to authenticate with IBM Quantum: {e}")
    
    def _mock_authenticate(self) -> bool:
        """Mock authentication for testing without IBM credentials."""
        self._service = None  # Mock service
        return True
    
    @circuit_breaker('ibm_backends', CircuitBreakerConfig())
    @retry(RetryPolicy(max_attempts=2))
    def get_backends(self, refresh: bool = False) -> List[QuantumBackend]:
        """Get IBM Quantum backends."""
        if self._backends and not refresh:
            return list(self._backends.values())
        
        if self._service is None:
            # Return mock backends for testing
            mock_backends = [
                QuantumBackend(
                    name="ibmq_qasm_simulator",
                    provider=ProviderType.IBM_QUANTUM,
                    backend_type="simulator",
                    num_qubits=32,
                    max_shots=8192,
                    cost_per_shot=0.0,
                    queue_length=0,
                    operational=True
                ),
                QuantumBackend(
                    name="ibm_brisbane",
                    provider=ProviderType.IBM_QUANTUM,
                    backend_type="qpu",
                    num_qubits=127,
                    max_shots=4096,
                    cost_per_shot=0.0015,
                    queue_length=25,
                    operational=True
                )
            ]
            
            self._backends = {b.name: b for b in mock_backends}
            return mock_backends
        
        # Real implementation would query IBM service
        try:
            backends = []
            for backend in self._service.backends():
                quantum_backend = QuantumBackend(
                    name=backend.name,
                    provider=ProviderType.IBM_QUANTUM,
                    backend_type="simulator" if "simulator" in backend.name else "qpu",
                    num_qubits=backend.configuration().n_qubits,
                    max_shots=backend.configuration().max_shots,
                    queue_length=backend.status().pending_jobs,
                    operational=backend.status().operational,
                    properties=backend.properties().to_dict() if backend.properties() else {}
                )
                backends.append(quantum_backend)
            
            self._backends = {b.name: b for b in backends}
            return backends
            
        except Exception as e:
            raise BackendConnectionError(f"Failed to get IBM backends: {e}")
    
    @timeout(300)
    def submit_job(
        self, 
        circuit: Any, 
        backend_name: str, 
        shots: int,
        **kwargs
    ) -> QuantumJob:
        """Submit job to IBM Quantum."""
        backend = self.get_backend(backend_name)
        if not backend:
            raise BackendConnectionError(f"Backend {backend_name} not found")
        
        # Generate mock job ID for testing
        import uuid
        job_id = f"ibm_job_{uuid.uuid4().hex[:8]}"
        
        estimated_cost = self.estimate_cost(backend_name, shots)
        
        job = QuantumJob(
            job_id=job_id,
            provider=ProviderType.IBM_QUANTUM,
            backend_name=backend_name,
            status=JobStatus.QUEUED,
            submitted_time=datetime.now(),
            shots=shots,
            estimated_cost=estimated_cost
        )
        
        self._jobs[job_id] = job
        return job
    
    def get_job_status(self, job_id: str) -> QuantumJob:
        """Get IBM job status."""
        if job_id not in self._jobs:
            raise TestExecutionError(f"Job {job_id} not found")
        
        job = self._jobs[job_id]
        
        # Mock status progression for demo
        if job.status == JobStatus.QUEUED:
            # Simulate job starting
            job.status = JobStatus.RUNNING
            job.start_time = datetime.now()
        elif job.status == JobStatus.RUNNING:
            # Simulate completion after some time
            if job.start_time and (datetime.now() - job.start_time).total_seconds() > 10:
                job.status = JobStatus.COMPLETED
                job.end_time = datetime.now()
        
        return job
    
    def get_job_result(self, job_id: str) -> TestResult:
        """Get IBM job results."""
        job = self.get_job_status(job_id)
        
        if job.status != JobStatus.COMPLETED:
            raise TestExecutionError(f"Job {job_id} is not completed (status: {job.status.value})")
        
        # Mock result for demo
        mock_counts = {'00': job.shots // 2, '11': job.shots // 2}
        
        return TestResult(
            counts=mock_counts,
            shots=job.shots,
            execution_time=3.5,
            backend_name=job.backend_name,
            metadata={
                'provider': 'IBM Quantum',
                'job_id': job_id,
                'queue_time': (job.start_time - job.submitted_time).total_seconds() if job.start_time else 0,
                'execution_time': (job.end_time - job.start_time).total_seconds() if job.end_time and job.start_time else 0
            }
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel IBM job."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False


class AWSBraketProvider(QuantumProvider):
    """AWS Braket provider implementation."""
    
    def __init__(self, credentials: ProviderCredentials):
        super().__init__(credentials)
        self._braket = None
    
    def authenticate(self) -> bool:
        """Authenticate with AWS Braket."""
        try:
            import boto3
            from braket.aws import AwsDevice
            
            # Mock authentication for demo
            self._braket = None
            return True
            
        except ImportError:
            warnings.warn("AWS Braket SDK not available, using mock provider")
            return True
    
    def get_backends(self, refresh: bool = False) -> List[QuantumBackend]:
        """Get AWS Braket backends."""
        # Mock backends for demo
        mock_backends = [
            QuantumBackend(
                name="SV1",
                provider=ProviderType.AWS_BRAKET,
                backend_type="simulator",
                num_qubits=34,
                max_shots=10000,
                cost_per_shot=0.00075,
                queue_length=0,
                operational=True
            ),
            QuantumBackend(
                name="Aria-1",
                provider=ProviderType.AWS_BRAKET,
                backend_type="qpu",
                num_qubits=25,
                max_shots=1000,
                cost_per_shot=0.01,
                queue_length=15,
                operational=True
            )
        ]
        
        self._backends = {b.name: b for b in mock_backends}
        return mock_backends
    
    def submit_job(
        self, 
        circuit: Any, 
        backend_name: str, 
        shots: int,
        **kwargs
    ) -> QuantumJob:
        """Submit job to AWS Braket."""
        # Mock implementation
        import uuid
        job_id = f"braket_job_{uuid.uuid4().hex[:8]}"
        
        estimated_cost = self.estimate_cost(backend_name, shots)
        
        job = QuantumJob(
            job_id=job_id,
            provider=ProviderType.AWS_BRAKET,
            backend_name=backend_name,
            status=JobStatus.QUEUED,
            submitted_time=datetime.now(),
            shots=shots,
            estimated_cost=estimated_cost
        )
        
        self._jobs[job_id] = job
        return job
    
    def get_job_status(self, job_id: str) -> QuantumJob:
        """Get Braket job status."""
        # Mock implementation similar to IBM
        if job_id not in self._jobs:
            raise TestExecutionError(f"Job {job_id} not found")
        
        return self._jobs[job_id]
    
    def get_job_result(self, job_id: str) -> TestResult:
        """Get Braket job results."""
        # Mock implementation
        job = self._jobs[job_id]
        mock_counts = {'0': job.shots // 2, '1': job.shots // 2}
        
        return TestResult(
            counts=mock_counts,
            shots=job.shots,
            execution_time=2.1,
            backend_name=job.backend_name,
            metadata={
                'provider': 'AWS Braket',
                'job_id': job_id
            }
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel Braket job."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False


class ProviderManager:
    """Central manager for quantum providers."""
    
    def __init__(self):
        self._providers: Dict[str, QuantumProvider] = {}
        self._default_provider: Optional[str] = None
    
    def register_provider(
        self, 
        name: str, 
        provider_type: ProviderType,
        credentials: ProviderCredentials
    ) -> QuantumProvider:
        """Register a new quantum provider."""
        # Create provider instance based on type
        if provider_type == ProviderType.IBM_QUANTUM:
            provider = IBMQuantumProvider(credentials)
        elif provider_type == ProviderType.AWS_BRAKET:
            provider = AWSBraketProvider(credentials)
        else:
            raise ConfigurationError(f"Unsupported provider type: {provider_type}")
        
        # Authenticate provider
        if not provider.authenticate():
            raise ConfigurationError(f"Failed to authenticate provider {name}")
        
        self._providers[name] = provider
        
        # Set as default if first provider
        if not self._default_provider:
            self._default_provider = name
        
        return provider
    
    def get_provider(self, name: Optional[str] = None) -> Optional[QuantumProvider]:
        """Get provider by name, or default provider."""
        provider_name = name or self._default_provider
        return self._providers.get(provider_name)
    
    def get_all_backends(self) -> Dict[str, List[QuantumBackend]]:
        """Get backends from all registered providers."""
        all_backends = {}
        for name, provider in self._providers.items():
            try:
                backends = provider.get_backends()
                all_backends[name] = backends
            except Exception as e:
                warnings.warn(f"Failed to get backends from {name}: {e}")
        
        return all_backends
    
    def find_best_backend(
        self, 
        min_qubits: int = 1, 
        prefer_simulators: bool = True,
        max_cost: Optional[float] = None
    ) -> Tuple[str, QuantumBackend]:
        """Find the best backend across all providers."""
        best_backend = None
        best_provider = None
        best_score = float('inf')
        
        for provider_name, provider in self._providers.items():
            try:
                backends = provider.get_backends()
                
                for backend in backends:
                    if backend.num_qubits < min_qubits or not backend.operational:
                        continue
                    
                    if max_cost and backend.cost_per_shot > max_cost:
                        continue
                    
                    # Calculate score (lower is better)
                    score = backend.queue_length
                    
                    if prefer_simulators and backend.backend_type == "simulator":
                        score -= 1000  # Strong preference for simulators
                    
                    if score < best_score:
                        best_score = score
                        best_backend = backend
                        best_provider = provider_name
                        
            except Exception as e:
                warnings.warn(f"Error evaluating backends from {provider_name}: {e}")
        
        if not best_backend or not best_provider:
            raise BackendConnectionError("No suitable backend found")
        
        return best_provider, best_backend


# Global provider manager
_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager