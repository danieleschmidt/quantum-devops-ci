"""
Quantum DevOps CI - Python testing framework for quantum computing pipelines.

This package provides tools for noise-aware testing, hardware resource management,
and CI/CD integration for quantum computing workflows compatible with Qiskit,
Cirq, PennyLane, and other quantum frameworks.
"""

# Import only stable modules for now
from .exceptions import *
from .validation import SecurityValidator, ConfigValidator, QuantumCircuitValidator
from .security import SecurityContext, SecurityManager
from .caching import MemoryCache, DiskCache, MultiLevelCache, CacheManager

# Optional imports with fallbacks
try:
    from .testing import NoiseAwareTestBase
    NoiseAwareTest = NoiseAwareTestBase  # Backward compatibility
except ImportError:
    NoiseAwareTest = NoiseAwareTestBase = None

try:
    from .quantum_fixtures import quantum_fixture
except ImportError:
    quantum_fixture = None

try:
    from .linting import QuantumLinter, PulseLinter
except ImportError:
    QuantumLinter = PulseLinter = None

try:
    from .scheduling import QuantumJobScheduler
except ImportError:
    QuantumJobScheduler = None

try:
    from .monitoring import QuantumCIMonitor
except ImportError:
    QuantumCIMonitor = None

try:
    from .cost import CostOptimizer
except ImportError:
    CostOptimizer = None

try:
    from .concurrency import ConcurrentExecutor
except ImportError:
    ConcurrentExecutor = None

try:
    from .autoscaling import AutoScalingManager
except ImportError:
    AutoScalingManager = None

__version__ = "1.0.0"
__author__ = "Quantum DevOps Community"
__email__ = "community@quantum-devops.org"

# Public API
__all__ = [
    # Testing framework
    "NoiseAwareTest",
    "NoiseAwareTestBase", 
    "quantum_fixture",
    
    # Linting and validation
    "QuantumLinter", 
    "PulseLinter",
    
    # Resource management
    "QuantumJobScheduler",
    "CostOptimizer",
    
    # Monitoring and metrics
    "QuantumCIMonitor",
    
    # Deployment and testing
    "QuantumABTest",
    
    # Core infrastructure
    "SecurityContext",
    "SecurityManager",
    "CacheManager",
]

# Framework compatibility check
def check_framework_availability():
    """Check which quantum frameworks are available."""
    available_frameworks = {}
    
    try:
        import qiskit
        available_frameworks['qiskit'] = qiskit.__version__
    except ImportError:
        available_frameworks['qiskit'] = None
    
    try:
        import cirq
        available_frameworks['cirq'] = cirq.__version__
    except ImportError:
        available_frameworks['cirq'] = None
    
    try:
        import pennylane
        available_frameworks['pennylane'] = pennylane.__version__
    except ImportError:
        available_frameworks['pennylane'] = None
    
    try:
        import braket
        available_frameworks['braket'] = braket.__version__
    except ImportError:
        available_frameworks['braket'] = None
    
    return available_frameworks

# Initialize framework compatibility on import
AVAILABLE_FRAMEWORKS = check_framework_availability()