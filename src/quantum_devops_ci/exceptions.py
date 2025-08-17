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


class ValidationError(QuantumDevOpsError):
    """Raised when input validation fails."""
    pass


class SecurityError(QuantumDevOpsError):
    """Raised when security validation fails."""
    pass


class CircuitValidationError(ValidationError):
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


# Generation 4 Intelligence Exceptions

class OptimizationError(QuantumDevOpsError):
    """Raised when quantum circuit optimization fails."""
    pass


class ModelTrainingError(QuantumDevOpsError):
    """Raised when ML model training fails."""
    pass


class PredictionError(QuantumDevOpsError):
    """Raised when ML prediction fails."""
    pass


class InsufficientDataError(QuantumDevOpsError):
    """Raised when insufficient data is available for analysis."""
    pass


class QECError(QuantumDevOpsError):
    """Raised when quantum error correction fails."""
    pass


class SyndromeDecodeError(QECError):
    """Raised when error syndrome decoding fails."""
    pass


class LogicalOperationError(QECError):
    """Raised when logical qubit operation fails."""
    pass


class StatisticalTestError(QuantumDevOpsError):
    """Raised when statistical test execution fails."""
    pass


class ModelValidationError(QuantumDevOpsError):
    """Raised when model validation fails."""
    pass


class SovereigntyViolationError(QuantumDevOpsError):
    """Raised when quantum sovereignty rules are violated."""
    pass


class ExportControlError(QuantumDevOpsError):
    """Raised when export control regulations are violated."""
    pass


class ComplianceError(QuantumDevOpsError):
    """Raised when compliance requirements are not met."""
    pass


# Research Framework Exceptions

class QuantumResearchError(QuantumDevOpsError):
    """Raised when quantum research operations fail."""
    pass


class ExperimentError(QuantumResearchError):
    """Raised when quantum experiment execution fails."""
    pass


class StatisticalAnalysisError(QuantumResearchError):
    """Raised when statistical analysis fails."""
    pass


class HypothesisValidationError(QuantumResearchError):
    """Raised when hypothesis validation fails."""
    pass


# Alias for backward compatibility
QuantumValidationError = ValidationError
QuantumSecurityError = SecurityError
QuantumTimeoutError = QuantumDevOpsError
QuantumResourceError = ResourceExhaustionError
QuantumTestError = TestExecutionError