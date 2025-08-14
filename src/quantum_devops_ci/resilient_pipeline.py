"""
Resilient Pipeline Framework for Quantum DevOps CI/CD.

Advanced error handling, failure recovery, circuit breakers, retry strategies,
and comprehensive validation for production quantum computing workflows.
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Union
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failures detected, circuit is open
    HALF_OPEN = auto() # Testing if service has recovered


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    WARNING = "warning"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: str
    component: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitoring_window: float = 300.0  # 5 minutes
    max_concurrent_calls: int = 100


@dataclass
class ValidationRule:
    """Validation rule definition."""
    rule_id: str
    name: str
    description: str
    validator: Callable
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    enabled: bool = True
    timeout: float = 30.0


class ResilientPipeline:
    """
    Resilient Pipeline Framework for Quantum DevOps CI/CD.
    
    Features:
    - Circuit breaker pattern for fault tolerance
    - Intelligent retry strategies with adaptive backoff
    - Comprehensive error handling and recovery
    - Bulkhead isolation for component failures
    - Health monitoring and alerting
    - Graceful degradation strategies
    """
    
    def __init__(self, name: str = "quantum_pipeline"):
        """Initialize resilient pipeline."""
        self.name = name
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.component_health: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.metrics: Dict[str, int] = defaultdict(int)
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default retry and circuit breaker configurations."""
        
        # Default retry configurations for different components
        self.retry_configs.update({
            "circuit_compilation": RetryConfig(
                max_attempts=2,
                base_delay=0.5,
                strategy=RetryStrategy.LINEAR_BACKOFF
            ),
            "quantum_simulation": RetryConfig(
                max_attempts=3,
                base_delay=2.0,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_delay=30.0
            ),
            "hardware_execution": RetryConfig(
                max_attempts=5,
                base_delay=5.0,
                strategy=RetryStrategy.ADAPTIVE,
                max_delay=300.0
            ),
            "cost_calculation": RetryConfig(
                max_attempts=2,
                base_delay=1.0,
                strategy=RetryStrategy.FIXED_DELAY
            ),
            "security_validation": RetryConfig(
                max_attempts=1,  # Security checks should not be retried aggressively
                base_delay=0.5,
                strategy=RetryStrategy.FIXED_DELAY
            )
        })
        
        # Initialize circuit breakers for critical components
        for component in ["quantum_simulation", "hardware_execution", "cost_optimization"]:
            self.circuit_breakers[component] = CircuitBreaker(
                name=component,
                config=CircuitBreakerConfig()
            )
    
    async def execute_with_resilience(
        self, 
        operation: Callable,
        component: str,
        context: Dict[str, Any] = None,
        custom_retry_config: Optional[RetryConfig] = None
    ) -> Any:
        """
        Execute operation with full resilience features.
        
        Args:
            operation: Async callable to execute
            component: Component name for configuration lookup
            context: Execution context
            custom_retry_config: Override default retry configuration
            
        Returns:
            Operation result
            
        Raises:
            ResilientPipelineError: When all resilience mechanisms are exhausted
        """
        if context is None:
            context = {}
        
        correlation_id = context.get("correlation_id", f"{component}_{int(time.time())}")
        start_time = time.time()
        
        logger.info(f"Starting resilient execution for {component} (ID: {correlation_id})")
        
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(component)
            if circuit_breaker and not circuit_breaker.can_execute():
                raise CircuitBreakerOpenError(f"Circuit breaker open for {component}")
            
            # Get retry configuration
            retry_config = custom_retry_config or self.retry_configs.get(
                component, 
                RetryConfig()
            )
            
            # Execute with retry logic
            result = await self._execute_with_retry(
                operation, 
                component, 
                retry_config, 
                context,
                correlation_id
            )
            
            # Record successful execution
            execution_time = time.time() - start_time
            self._record_success(component, execution_time, correlation_id)
            
            if circuit_breaker:
                circuit_breaker.record_success()
            
            logger.info(f"Resilient execution completed for {component} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_context = self._create_error_context(e, component, context, correlation_id)
            
            # Record failure
            self._record_failure(component, error_context, execution_time)
            
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            # Attempt recovery if possible
            recovery_result = await self._attempt_recovery(error_context, operation, context)
            if recovery_result is not None:
                logger.info(f"Recovery successful for {component}")
                return recovery_result
            
            logger.error(f"Resilient execution failed for {component} after {execution_time:.2f}s")
            raise ResilientPipelineError(f"All resilience mechanisms exhausted for {component}") from e
    
    async def _execute_with_retry(
        self,
        operation: Callable,
        component: str,
        retry_config: RetryConfig,
        context: Dict[str, Any],
        correlation_id: str
    ) -> Any:
        """Execute operation with retry logic."""
        
        last_exception = None
        retry_delays = self._calculate_retry_delays(retry_config)
        
        for attempt in range(retry_config.max_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retry_config.max_attempts} for {component}")
                
                # Add retry context
                execution_context = {
                    **context,
                    "attempt": attempt + 1,
                    "max_attempts": retry_config.max_attempts,
                    "correlation_id": correlation_id
                }
                
                result = await operation(execution_context)
                
                if attempt > 0:
                    logger.info(f"Operation {component} succeeded on attempt {attempt + 1}")
                
                self.metrics[f"{component}_success_attempts"] += attempt + 1
                return result
                
            except Exception as e:
                last_exception = e
                self.metrics[f"{component}_failed_attempts"] += 1
                
                # Check if this exception type should trigger retry
                if not self._should_retry(e, retry_config):
                    logger.warning(f"Exception {type(e).__name__} not configured for retry")
                    raise
                
                # Don't sleep after the last attempt
                if attempt < retry_config.max_attempts - 1:
                    delay = retry_delays[attempt]
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {component}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted for {component}")
        
        # All retries exhausted
        self.metrics[f"{component}_total_failures"] += 1
        raise last_exception
    
    def _calculate_retry_delays(self, config: RetryConfig) -> List[float]:
        """Calculate retry delays based on strategy."""
        delays = []
        
        for attempt in range(config.max_attempts - 1):  # No delay after last attempt
            if config.strategy == RetryStrategy.FIXED_DELAY:
                delay = config.base_delay
                
            elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
                delay = config.base_delay * (attempt + 1)
                
            elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                delay = config.base_delay * (config.backoff_multiplier ** attempt)
                
            elif config.strategy == RetryStrategy.FIBONACCI:
                # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...
                if attempt <= 1:
                    delay = config.base_delay
                else:
                    fib_prev = delays[-2] if len(delays) >= 2 else config.base_delay
                    fib_current = delays[-1] if delays else config.base_delay
                    delay = fib_prev + fib_current
                    
            elif config.strategy == RetryStrategy.ADAPTIVE:
                # Adaptive strategy adjusts based on historical success rates
                success_rate = self._get_component_success_rate(config)
                if success_rate > 0.8:
                    delay = config.base_delay * 0.5  # Faster retry for reliable components
                elif success_rate < 0.3:
                    delay = config.base_delay * 3.0  # Slower retry for unreliable components
                else:
                    delay = config.base_delay * (config.backoff_multiplier ** attempt)
            else:
                delay = config.base_delay
            
            # Apply jitter if enabled
            if config.jitter:
                import random
                jitter_factor = 0.1  # ±10% jitter
                jitter = random.uniform(-jitter_factor, jitter_factor)
                delay = delay * (1 + jitter)
            
            # Ensure delay doesn't exceed maximum
            delay = min(delay, config.max_delay)
            delays.append(delay)
        
        return delays
    
    def _should_retry(self, exception: Exception, config: RetryConfig) -> bool:
        """Determine if exception should trigger retry."""
        # Check if exception type is in retry list
        for retry_exception_type in config.retry_on_exceptions:
            if isinstance(exception, retry_exception_type):
                return True
        
        # Don't retry certain critical exceptions
        if isinstance(exception, (KeyboardInterrupt, SystemExit, MemoryError)):
            return False
        
        # Don't retry validation errors
        if isinstance(exception, ValidationError):
            return False
        
        return False
    
    def _get_component_success_rate(self, config: RetryConfig) -> float:
        """Get historical success rate for adaptive retry strategy."""
        # Simplified implementation - in production this would analyze historical data
        return 0.7  # Default moderate success rate
    
    async def _attempt_recovery(
        self, 
        error_context: ErrorContext, 
        operation: Callable, 
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Attempt to recover from failure using various strategies."""
        
        component = error_context.component
        
        # Strategy 1: Fallback to alternative implementation
        fallback_result = await self._try_fallback(component, context)
        if fallback_result is not None:
            logger.info(f"Fallback recovery successful for {component}")
            return fallback_result
        
        # Strategy 2: Degraded mode execution
        if component in ["quantum_simulation", "performance_benchmarks"]:
            degraded_result = await self._try_degraded_execution(component, context)
            if degraded_result is not None:
                logger.info(f"Degraded mode recovery successful for {component}")
                return degraded_result
        
        # Strategy 3: Cache/previous result recovery
        cached_result = await self._try_cache_recovery(component, context)
        if cached_result is not None:
            logger.info(f"Cache recovery successful for {component}")
            return cached_result
        
        # No recovery possible
        return None
    
    async def _try_fallback(self, component: str, context: Dict[str, Any]) -> Optional[Any]:
        """Try fallback implementation."""
        fallback_strategies = {
            "quantum_simulation": self._fallback_to_simple_simulator,
            "cost_calculation": self._fallback_to_basic_cost_model,
            "optimization": self._fallback_to_basic_optimization
        }
        
        if component in fallback_strategies:
            try:
                return await fallback_strategies[component](context)
            except Exception as e:
                logger.warning(f"Fallback failed for {component}: {e}")
        
        return None
    
    async def _fallback_to_simple_simulator(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic simulation with reduced fidelity."""
        return {
            "simulation_result": "basic_simulation_completed",
            "fidelity": 0.85,  # Reduced but acceptable
            "mode": "fallback",
            "warning": "Using simplified simulation due to primary failure"
        }
    
    async def _fallback_to_basic_cost_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic cost calculation."""
        gate_count = context.get("gate_count", 100)
        basic_cost = gate_count * 0.01  # $0.01 per gate (simplified)
        
        return {
            "estimated_cost": basic_cost,
            "currency": "USD",
            "mode": "fallback",
            "accuracy": "low",
            "warning": "Using simplified cost model"
        }
    
    async def _fallback_to_basic_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic circuit optimization."""
        return {
            "optimization_result": "basic_optimization_applied",
            "improvement": 0.15,  # 15% improvement
            "mode": "fallback",
            "warning": "Using basic optimization due to advanced optimizer failure"
        }
    
    async def _try_degraded_execution(self, component: str, context: Dict[str, Any]) -> Optional[Any]:
        """Try degraded mode execution with reduced functionality."""
        if component == "quantum_simulation":
            # Reduce simulation complexity
            context_degraded = {**context, "shots": min(context.get("shots", 1000), 100)}
            return await self._fallback_to_simple_simulator(context_degraded)
        
        return None
    
    async def _try_cache_recovery(self, component: str, context: Dict[str, Any]) -> Optional[Any]:
        """Try to recover from cache or previous results."""
        # In a real implementation, this would check Redis, database, or local cache
        # For demo purposes, return None (no cache available)
        return None
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule to the pipeline."""
        self.validation_rules[rule.rule_id] = rule
        logger.info(f"Added validation rule: {rule.name}")
    
    async def validate_input(self, data: Dict[str, Any], component: str) -> List[ErrorContext]:
        """Validate input data using configured rules."""
        errors = []
        
        for rule_id, rule in self.validation_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Execute validation with timeout
                result = await asyncio.wait_for(
                    rule.validator(data, component),
                    timeout=rule.timeout
                )
                
                if not result.get("valid", True):
                    error_context = ErrorContext(
                        error_type="validation_error",
                        error_message=result.get("message", f"Validation failed: {rule.name}"),
                        severity=rule.severity,
                        timestamp=datetime.now(),
                        stack_trace="",
                        component=component,
                        context_data={
                            "rule_id": rule_id,
                            "rule_name": rule.name,
                            "validation_details": result
                        }
                    )
                    errors.append(error_context)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Validation rule {rule.name} timed out")
                errors.append(ErrorContext(
                    error_type="validation_timeout",
                    error_message=f"Validation rule {rule.name} exceeded timeout",
                    severity=ErrorSeverity.WARNING,
                    timestamp=datetime.now(),
                    stack_trace="",
                    component=component
                ))
                
            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                errors.append(ErrorContext(
                    error_type="validation_exception",
                    error_message=f"Validation rule {rule.name} raised exception: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    timestamp=datetime.now(),
                    stack_trace=traceback.format_exc(),
                    component=component
                ))
        
        return errors
    
    def _create_error_context(
        self, 
        exception: Exception, 
        component: str, 
        context: Dict[str, Any],
        correlation_id: str
    ) -> ErrorContext:
        """Create comprehensive error context."""
        
        # Determine error severity
        severity = ErrorSeverity.MEDIUM
        if isinstance(exception, (MemoryError, SystemExit)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            severity = ErrorSeverity.HIGH
        elif isinstance(exception, ValidationError):
            severity = ErrorSeverity.WARNING
        
        return ErrorContext(
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=severity,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            component=component,
            context_data=context,
            correlation_id=correlation_id,
            recovery_suggestions=self._generate_recovery_suggestions(exception, component)
        )
    
    def _generate_recovery_suggestions(self, exception: Exception, component: str) -> List[str]:
        """Generate recovery suggestions based on error type and component."""
        suggestions = []
        
        if isinstance(exception, ConnectionError):
            suggestions.extend([
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Consider using fallback provider"
            ])
        
        elif isinstance(exception, TimeoutError):
            suggestions.extend([
                "Increase timeout configuration",
                "Reduce workload complexity",
                "Check system resource utilization"
            ])
        
        elif isinstance(exception, MemoryError):
            suggestions.extend([
                "Reduce batch size or circuit complexity",
                "Consider streaming processing",
                "Check for memory leaks"
            ])
        
        # Component-specific suggestions
        if component == "quantum_simulation":
            suggestions.extend([
                "Try reducing the number of shots",
                "Use a simpler noise model",
                "Consider approximate simulation methods"
            ])
        
        elif component == "circuit_compilation":
            suggestions.extend([
                "Simplify circuit structure",
                "Check gate compatibility with target backend",
                "Verify circuit parameters are valid"
            ])
        
        return suggestions
    
    def _record_success(self, component: str, execution_time: float, correlation_id: str):
        """Record successful execution metrics."""
        self.metrics[f"{component}_total_successes"] += 1
        self.metrics[f"{component}_total_executions"] += 1
        
        # Update component health
        self.component_health[component].update({
            "last_success": datetime.now(),
            "avg_execution_time": execution_time,
            "status": "healthy"
        })
    
    def _record_failure(self, component: str, error_context: ErrorContext, execution_time: float):
        """Record failure metrics and error history."""
        self.metrics[f"{component}_total_failures"] += 1
        self.metrics[f"{component}_total_executions"] += 1
        
        # Add to error history
        self.error_history.append(error_context)
        
        # Update component health
        self.component_health[component].update({
            "last_failure": datetime.now(),
            "last_error": error_context.error_message,
            "status": "degraded" if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else "warning"
        })
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get comprehensive pipeline health status."""
        total_executions = sum(
            self.metrics.get(f"{comp}_total_executions", 0) 
            for comp in self.component_health.keys()
        )
        
        total_failures = sum(
            self.metrics.get(f"{comp}_total_failures", 0) 
            for comp in self.component_health.keys()
        )
        
        overall_success_rate = 1.0 - (total_failures / total_executions) if total_executions > 0 else 1.0
        
        # Analyze recent errors
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        critical_errors = [
            err for err in recent_errors 
            if err.severity == ErrorSeverity.CRITICAL
        ]
        
        # Determine overall health status
        if critical_errors or overall_success_rate < 0.5:
            health_status = "critical"
        elif overall_success_rate < 0.8:
            health_status = "degraded"
        elif overall_success_rate < 0.95:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "overall_status": health_status,
            "success_rate": overall_success_rate,
            "total_executions": total_executions,
            "total_failures": total_failures,
            "recent_errors_count": len(recent_errors),
            "critical_errors_count": len(critical_errors),
            "component_health": dict(self.component_health),
            "circuit_breaker_status": {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            },
            "metrics": dict(self.metrics)
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis and patterns."""
        if not self.error_history:
            return {"status": "no_errors", "analysis": {}}
        
        # Group errors by type and component
        error_by_type = defaultdict(int)
        error_by_component = defaultdict(int)
        error_by_severity = defaultdict(int)
        
        for error in self.error_history:
            error_by_type[error.error_type] += 1
            error_by_component[error.component] += 1
            error_by_severity[error.severity.value] += 1
        
        # Find most problematic components
        most_errors = max(error_by_component.items(), key=lambda x: x[1]) if error_by_component else ("none", 0)
        
        # Recent error trend
        now = datetime.now()
        recent_1h = sum(1 for e in self.error_history if e.timestamp > now - timedelta(hours=1))
        recent_24h = sum(1 for e in self.error_history if e.timestamp > now - timedelta(hours=24))
        
        return {
            "total_errors": len(self.error_history),
            "error_distribution": {
                "by_type": dict(error_by_type),
                "by_component": dict(error_by_component),
                "by_severity": dict(error_by_severity)
            },
            "most_problematic_component": most_errors[0],
            "recent_trends": {
                "last_1_hour": recent_1h,
                "last_24_hours": recent_24h
            },
            "top_recommendations": self._get_top_recommendations()
        }
    
    def _get_top_recommendations(self) -> List[str]:
        """Get top recommendations based on error patterns."""
        recommendations = []
        
        # Analyze error patterns and suggest improvements
        if self.error_history:
            recent_errors = [
                e for e in self.error_history 
                if e.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if len(recent_errors) > 10:
                recommendations.append("High error rate detected - consider reviewing system configuration")
            
            # Check for timeout patterns
            timeout_errors = [e for e in recent_errors if "timeout" in e.error_type.lower()]
            if len(timeout_errors) > 3:
                recommendations.append("Multiple timeout errors - consider increasing timeout values or reducing workload")
            
            # Check for connection issues
            connection_errors = [e for e in recent_errors if "connection" in e.error_type.lower()]
            if len(connection_errors) > 2:
                recommendations.append("Connection issues detected - check network connectivity and service availability")
        
        return recommendations


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.current_calls = 0
        
        # Sliding window for monitoring
        self.call_history = deque(maxlen=100)
    
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution."""
        now = time.time()
        
        # Check concurrent call limit
        if self.current_calls >= self.config.max_concurrent_calls:
            return False
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if enough time has passed to try recovery
            if (self.last_failure_time and 
                now - self.last_failure_time > self.config.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited calls to test recovery
            return self.success_count < self.config.success_threshold
        
        return False
    
    def record_success(self):
        """Record successful execution."""
        self.current_calls = max(0, self.current_calls - 1)
        self.call_history.append(("success", time.time()))
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED state")
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution."""
        self.current_calls = max(0, self.current_calls - 1)
        self.call_history.append(("failure", time.time()))
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} reopened after failed recovery attempt")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        now = time.time()
        
        # Calculate recent metrics
        recent_calls = [
            call for call in self.call_history 
            if now - call[1] < self.config.monitoring_window
        ]
        
        total_calls = len(recent_calls)
        failed_calls = len([call for call in recent_calls if call[0] == "failure"])
        success_rate = 1.0 - (failed_calls / total_calls) if total_calls > 0 else 1.0
        
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "current_calls": self.current_calls,
            "recent_success_rate": success_rate,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "max_concurrent_calls": self.config.max_concurrent_calls
            }
        }


# Custom exception classes
class ResilientPipelineError(Exception):
    """Base exception for resilient pipeline errors."""
    pass


class CircuitBreakerOpenError(ResilientPipelineError):
    """Raised when circuit breaker is open."""
    pass


class ValidationError(ResilientPipelineError):
    """Raised when validation fails."""
    pass


# Validation rule examples
async def validate_circuit_parameters(data: Dict[str, Any], component: str) -> Dict[str, Any]:
    """Validate quantum circuit parameters."""
    errors = []
    
    # Check gate count
    gate_count = data.get("gate_count", 0)
    if gate_count > 1000:
        errors.append("Gate count exceeds recommended maximum of 1000")
    
    # Check circuit depth
    depth = data.get("circuit_depth", 0)
    if depth > 200:
        errors.append("Circuit depth exceeds recommended maximum of 200")
    
    # Check qubit count
    qubit_count = data.get("qubit_count", 0)
    if qubit_count > 50:
        errors.append("Qubit count exceeds available hardware limits")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": []
    }


async def validate_security_context(data: Dict[str, Any], component: str) -> Dict[str, Any]:
    """Validate security-related parameters."""
    errors = []
    warnings = []
    
    # Check for potential secrets in data
    sensitive_keys = ["password", "token", "key", "secret", "credential"]
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            if isinstance(value, str) and len(value) > 10:
                errors.append(f"Potential secret detected in field: {key}")
    
    # Check for dangerous operations
    dangerous_operations = data.get("operations", [])
    if "system_call" in dangerous_operations:
        warnings.append("System call operations detected - ensure proper sandboxing")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


# Convenience function for easy integration
async def create_resilient_pipeline(name: str = "quantum_pipeline") -> ResilientPipeline:
    """Create a fully configured resilient pipeline."""
    pipeline = ResilientPipeline(name)
    
    # Add default validation rules
    pipeline.add_validation_rule(ValidationRule(
        rule_id="circuit_validation",
        name="Circuit Parameter Validation",
        description="Validates quantum circuit parameters are within acceptable ranges",
        validator=validate_circuit_parameters,
        severity=ErrorSeverity.HIGH
    ))
    
    pipeline.add_validation_rule(ValidationRule(
        rule_id="security_validation",
        name="Security Context Validation",
        description="Validates no sensitive data is exposed",
        validator=validate_security_context,
        severity=ErrorSeverity.CRITICAL
    ))
    
    return pipeline