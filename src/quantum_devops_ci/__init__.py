"""
Quantum DevOps CI - Python testing framework for quantum computing pipelines.

This package provides tools for noise-aware testing, hardware resource management,
and CI/CD integration for quantum computing workflows compatible with Qiskit,
Cirq, PennyLane, and other quantum frameworks.
"""

from .testing import NoiseAwareTest, quantum_fixture
from .linting import QuantumLinter, PulseLinter
from .scheduling import QuantumJobScheduler
from .monitoring import QuantumCIMonitor
from .cost import CostOptimizer
# from .deployment import QuantumABTest  # Temporarily commented out due to syntax error

__version__ = "1.0.0"
__author__ = "Quantum DevOps Community"
__email__ = "community@quantum-devops.org"

# Public API
__all__ = [
    # Testing framework
    "NoiseAwareTest",
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
    # "QuantumABTest",  # Temporarily commented out
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