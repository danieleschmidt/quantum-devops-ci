"""
Resilience and error recovery framework for quantum DevOps CI/CD.

This module provides circuit breakers, retries, fallbacks, and other
resilience patterns specifically designed for quantum computing workflows.
"""

import time
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .exceptions import (
    TestExecutionError, BackendConnectionError, ResourceExhaustionError,
    NoiseModelError, CircuitValidationError, SecurityError, ValidationError
)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_backoff: bool = True
    jitter: bool = True
    
    # Exceptions that should trigger retry
    retriable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        BackendConnectionError,
        ResourceExhaustionError,
        TestExecutionError
    ])
    
    # Exceptions that should NOT be retried
    non_retriable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        CircuitValidationError,
        SecurityError,
        ValidationError
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Open after this many failures
    recovery_timeout: float = 60.0  # seconds to wait before testing
    success_threshold: int = 2  # Successes needed to close from half-open
    
    # Exceptions that count as failures
    failure_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        BackendConnectionError,
        ResourceExhaustionError,
        TestExecutionError
    ])


class CircuitBreaker:
    """Circuit breaker for quantum service calls."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker logic."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} moving to half-open")
            else:
                self.logger.warning(f"Circuit breaker {self.name} is open, blocking call")
                raise BackendConnectionError(
                    f"Circuit breaker {self.name} is open. "
                    f"Last failure: {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            if any(isinstance(e, exc_type) for exc_type in self.config.failure_exceptions):
                self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if not self.last_failure_time:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} closed after recovery")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.logger.warning(f"Circuit breaker {self.name} opened after half-open failure")
        
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(
                f"Circuit breaker {self.name} opened after {self.failure_count} failures"
            )
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger.info(f"Circuit breaker {self.name} manually reset")


class RetryHandler:
    """Advanced retry handler with exponential backoff and jitter."""
    
    def __init__(self, policy: RetryPolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry(func, *args, **kwargs)
        return wrapper
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.policy.max_attempts):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(
                        f"Retrying {func.__name__} (attempt {attempt + 1}/{self.policy.max_attempts}) "
                        f"after {delay:.2f}s delay"
                    )
                    time.sleep(delay)
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception should be retried
                if not self._should_retry(e):
                    self.logger.info(f"Not retrying {func.__name__}: {type(e).__name__} is not retriable")
                    raise
                
                # Don't sleep on the last attempt
                if attempt == self.policy.max_attempts - 1:
                    self.logger.error(
                        f"Final attempt failed for {func.__name__}: {e}"
                    )
                    raise
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                )
        
        # This should not be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Check non-retriable exceptions first
        if any(isinstance(exception, exc_type) for exc_type in self.policy.non_retriable_exceptions):
            return False
        
        # Check retriable exceptions
        return any(isinstance(exception, exc_type) for exc_type in self.policy.retriable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.policy.exponential_backoff:
            delay = self.policy.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.policy.base_delay
        
        # Apply maximum delay
        delay = min(delay, self.policy.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.policy.jitter:
            import random
            jitter = random.uniform(0.1, 0.9) * delay
            delay = delay * 0.5 + jitter
        
        return delay


class FallbackHandler:
    """Fallback execution for failed quantum operations."""
    
    def __init__(self, fallback_func: Callable, conditions: Optional[List[Type[Exception]]] = None):
        self.fallback_func = fallback_func
        self.conditions = conditions or [Exception]  # Fallback on any exception by default
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply fallback logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self._should_fallback(e):
                    self.logger.warning(
                        f"Primary function {func.__name__} failed ({type(e).__name__}), "
                        f"executing fallback"
                    )
                    return self.fallback_func(*args, **kwargs)
                else:
                    raise
        return wrapper
    
    def _should_fallback(self, exception: Exception) -> bool:
        """Determine if exception should trigger fallback."""
        return any(isinstance(exception, condition) for condition in self.conditions)


class TimeoutHandler:
    """Timeout wrapper for long-running quantum operations."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply timeout logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise ResourceExhaustionError(
                    f"Function {func.__name__} timed out after {self.timeout_seconds}s"
                )
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Clean up timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper


class QuantumResilienceManager:
    """Central manager for quantum operation resilience."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        config = config or CircuitBreakerConfig()
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        self.logger.info(f"Created circuit breaker: {name}")
        return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def create_retry_policy(self, name: str, policy: Optional[RetryPolicy] = None) -> RetryPolicy:
        """Create and register a retry policy."""
        policy = policy or RetryPolicy()
        self.retry_policies[name] = policy
        self.logger.info(f"Created retry policy: {name}")
        return policy
    
    def resilient_quantum_call(
        self, 
        func: Callable, 
        circuit_breaker_name: Optional[str] = None,
        retry_policy_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        fallback_func: Optional[Callable] = None,
        *args, 
        **kwargs
    ) -> Any:
        """Execute quantum function with full resilience patterns."""
        # Apply decorators in order: timeout -> retry -> circuit breaker -> fallback
        resilient_func = func
        
        # Apply timeout if specified
        if timeout_seconds:
            resilient_func = TimeoutHandler(timeout_seconds)(resilient_func)
        
        # Apply retry policy if specified
        if retry_policy_name and retry_policy_name in self.retry_policies:
            retry_handler = RetryHandler(self.retry_policies[retry_policy_name])
            resilient_func = retry_handler(resilient_func)
        
        # Apply circuit breaker if specified
        if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
            breaker = self.circuit_breakers[circuit_breaker_name]
            resilient_func = breaker(resilient_func)
        
        # Apply fallback if specified
        if fallback_func:
            fallback_handler = FallbackHandler(fallback_func)
            resilient_func = fallback_handler(resilient_func)
        
        # Execute with all resilience patterns applied
        return resilient_func(*args, **kwargs)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        status = {
            'circuit_breakers': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, breaker in self.circuit_breakers.items():
            status['circuit_breakers'][name] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count,
                'last_failure_time': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
            }
        
        return status
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers to closed state."""
        for name, breaker in self.circuit_breakers.items():
            breaker.reset()
        
        self.logger.info("Reset all circuit breakers")


# Global resilience manager instance
_resilience_manager: Optional[QuantumResilienceManager] = None


def get_resilience_manager() -> QuantumResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = QuantumResilienceManager()
    return _resilience_manager


# Convenience decorators using global manager
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker pattern."""
    manager = get_resilience_manager()
    if name not in manager.circuit_breakers:
        manager.create_circuit_breaker(name, config)
    return manager.circuit_breakers[name]


def retry(policy: Optional[RetryPolicy] = None):
    """Decorator to apply retry pattern."""
    policy = policy or RetryPolicy()
    return RetryHandler(policy)


def timeout(seconds: float):
    """Decorator to apply timeout pattern."""
    return TimeoutHandler(seconds)


def fallback(fallback_func: Callable, conditions: Optional[List[Type[Exception]]] = None):
    """Decorator to apply fallback pattern."""
    return FallbackHandler(fallback_func, conditions)
