"""
quantum-devops-ci: A DevOps toolkit for quantum computing pipelines.

Provides noise-aware testing, CI/CD templates, and resource estimation
for quantum software — no external dependencies required.
"""

from .circuit import QuantumCircuit, Gate
from .simulator import NoisySimulator
from .testing import NoiseAwareTestRunner, TestResult, CircuitTestReport
from .ci_template import CITemplate
from .resources import ResourceEstimator, ResourceReport

__version__ = "2.0.0"
__all__ = [
    "QuantumCircuit",
    "Gate",
    "NoisySimulator",
    "NoiseAwareTestRunner",
    "TestResult",
    "CircuitTestReport",
    "CITemplate",
    "ResourceEstimator",
    "ResourceReport",
]
