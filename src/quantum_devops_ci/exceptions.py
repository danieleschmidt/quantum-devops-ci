"""
Custom exceptions for quantum DevOps CI system.

This module defines custom exception classes for better error handling
and debugging across the quantum CI/CD pipeline.
"""

from typing import Optional, Dict, Any


class QuantumDevOpsError(Exception):
    """Base exception for quantum DevOps CI system."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class CircuitValidationError(QuantumDevOpsError):
    """Raised when quantum circuit validation fails."""
    pass


class TestExecutionError(QuantumDevOpsError):
    """Raised when quantum test execution fails."""
    pass


class NoiseModelError(QuantumDevOpsError):
    """Raised when noise model configuration is invalid."""
    pass


class BackendConnectionError(QuantumDevOpsError):
    """Raised when quantum backend connection fails."""
    pass


class ResourceExhaustionError(QuantumDevOpsError):
    """Raised when system resources are exhausted."""
    pass


class SchedulingError(QuantumDevOpsError):
    """Raised when job scheduling fails."""
    pass


class CostOptimizationError(QuantumDevOpsError):
    """Raised when cost optimization fails."""
    pass


class MonitoringError(QuantumDevOpsError):
    """Raised when monitoring system fails."""
    pass


class ConfigurationError(QuantumDevOpsError):
    """Raised when configuration is invalid."""
    pass


class SecurityError(QuantumDevOpsError):
    """Raised when security validation fails."""
    pass


class LintingError(QuantumDevOpsError):
    """Raised when circuit linting fails."""
    pass


class ProviderError(QuantumDevOpsError):
    """Raised when quantum provider interactions fail."""
    pass


class QuotaExceededError(QuantumDevOpsError):
    """Raised when usage quotas are exceeded."""
    pass


class AuthenticationError(QuantumDevOpsError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(QuantumDevOpsError):
    """Raised when authorization fails."""
    pass


class DeploymentError(QuantumDevOpsError):
    """Raised when deployment operations fail."""
    pass


class ValidationError(QuantumDevOpsError):
    """Raised when validation fails."""
    pass