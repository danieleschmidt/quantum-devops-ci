"""
Health monitoring and system diagnostics for quantum DevOps CI/CD.

This module provides comprehensive health checks, system monitoring,
and diagnostic capabilities for the quantum testing infrastructure.
"""

import time
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

from .resilience import get_resilience_manager
from .providers import get_provider_manager
from .database.connection import get_connection
from .exceptions import ResourceExhaustionError, BackendConnectionError


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_status': self.overall_status.value,
            'checks': [check.to_dict() for check in self.checks],
            'timestamp': self.timestamp.isoformat(),
            'summary': {
                'healthy': sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                'warning': sum(1 for c in self.checks if c.status == HealthStatus.WARNING),
                'critical': sum(1 for c in self.checks if c.status == HealthStatus.CRITICAL),
                'unknown': sum(1 for c in self.checks if c.status == HealthStatus.UNKNOWN),
                'total': len(self.checks)
            }
        }


class HealthChecker:
    """Base class for health checks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def check(self) -> HealthCheck:
        """Execute health check and return result."""
        start_time = time.time()
        
        try:
            status, message, details = self._execute_check()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name=self.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check {self.name} failed: {e}")
            
            return HealthCheck(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    def _execute_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError("Subclasses must implement _execute_check")


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database connectivity."""
    
    def __init__(self):
        super().__init__("database")
    
    def _execute_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check database connectivity and performance."""
        try:
            db = get_connection()
            
            # Test basic connectivity
            start_time = time.time()
            result = db.execute_query("SELECT 1 as test")
            query_time = (time.time() - start_time) * 1000
            
            if not result or result[0]['test'] != 1:
                return HealthStatus.CRITICAL, "Database query returned unexpected result", {}
            
            # Check database file size (for SQLite)
            details = {
                'query_time_ms': query_time,
                'database_type': db.config.database_type
            }
            
            if db.config.database_type == "sqlite":
                import os
                if os.path.exists(db.config.database_name):
                    size_bytes = os.path.getsize(db.config.database_name)
                    details['database_size_bytes'] = size_bytes
                    details['database_size_mb'] = size_bytes / (1024 * 1024)
            
            # Determine status based on query performance
            if query_time > 1000:  # > 1 second
                status = HealthStatus.WARNING
                message = f"Database responding slowly ({query_time:.1f}ms)"
            elif query_time > 100:  # > 100ms
                status = HealthStatus.WARNING
                message = f"Database response time elevated ({query_time:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy ({query_time:.1f}ms)"
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Database connection failed: {e}", {'error': str(e)}


class SystemResourcesHealthChecker(HealthChecker):
    """Health checker for system resources (CPU, memory, disk)."""
    
    def __init__(self):
        super().__init__("system_resources")
    
    def _execute_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system resource utilization."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
            
            # Determine overall health status
            issues = []
            status = HealthStatus.HEALTHY
            
            # Check CPU usage
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = HealthStatus.CRITICAL
            elif cpu_percent > 70:
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
                status = max(status, HealthStatus.WARNING)
            
            # Check memory usage
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = HealthStatus.CRITICAL
            elif memory.percent > 80:
                issues.append(f"Elevated memory usage: {memory.percent:.1f}%")
                status = max(status, HealthStatus.WARNING)
            
            # Check disk usage
            if disk.percent > 95:
                issues.append(f"Critical disk usage: {disk.percent:.1f}%")
                status = HealthStatus.CRITICAL
            elif disk.percent > 85:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
                status = max(status, HealthStatus.WARNING)
            
            # Generate message
            if issues:
                message = "; ".join(issues)
            else:
                message = f"System resources healthy (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%)"
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Unable to check system resources: {e}", {'error': str(e)}


class QuantumProvidersHealthChecker(HealthChecker):
    """Health checker for quantum provider connections."""
    
    def __init__(self):
        super().__init__("quantum_providers")
    
    def _execute_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check quantum provider connectivity and status."""
        try:
            provider_manager = get_provider_manager()
            
            if not provider_manager._providers:
                return HealthStatus.WARNING, "No quantum providers registered", {'provider_count': 0}
            
            provider_statuses = {}
            overall_status = HealthStatus.HEALTHY
            healthy_count = 0
            
            for name, provider in provider_manager._providers.items():
                try:
                    # Try to get backends as a connectivity test
                    backends = provider.get_backends()
                    operational_backends = [b for b in backends if b.operational]
                    
                    provider_statuses[name] = {
                        'status': 'healthy',
                        'total_backends': len(backends),
                        'operational_backends': len(operational_backends),
                        'provider_type': provider.provider_type.value
                    }
                    healthy_count += 1
                    
                except Exception as e:
                    provider_statuses[name] = {
                        'status': 'failed',
                        'error': str(e),
                        'provider_type': provider.provider_type.value
                    }
                    overall_status = HealthStatus.WARNING
            
            # Determine overall status
            if healthy_count == 0:
                overall_status = HealthStatus.CRITICAL
                message = "All quantum providers are unavailable"
            elif healthy_count < len(provider_manager._providers):
                overall_status = HealthStatus.WARNING
                message = f"{healthy_count}/{len(provider_manager._providers)} quantum providers healthy"
            else:
                message = f"All {healthy_count} quantum providers healthy"
            
            details = {
                'provider_count': len(provider_manager._providers),
                'healthy_count': healthy_count,
                'providers': provider_statuses
            }
            
            return overall_status, message, details
            
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Unable to check quantum providers: {e}", {'error': str(e)}


class CircuitBreakerHealthChecker(HealthChecker):
    """Health checker for circuit breaker states."""
    
    def __init__(self):
        super().__init__("circuit_breakers")
    
    def _execute_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check circuit breaker health status."""
        try:
            resilience_manager = get_resilience_manager()
            health_status = resilience_manager.get_health_status()
            
            circuit_breakers = health_status.get('circuit_breakers', {})
            
            if not circuit_breakers:
                return HealthStatus.HEALTHY, "No circuit breakers registered", {'breaker_count': 0}
            
            open_breakers = []
            half_open_breakers = []
            
            for name, breaker_info in circuit_breakers.items():
                if breaker_info['state'] == 'open':
                    open_breakers.append(name)
                elif breaker_info['state'] == 'half_open':
                    half_open_breakers.append(name)
            
            # Determine status
            if open_breakers:
                status = HealthStatus.CRITICAL
                message = f"Circuit breakers open: {', '.join(open_breakers)}"
            elif half_open_breakers:
                status = HealthStatus.WARNING
                message = f"Circuit breakers testing recovery: {', '.join(half_open_breakers)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {len(circuit_breakers)} circuit breakers closed"
            
            details = {
                'total_breakers': len(circuit_breakers),
                'open_breakers': open_breakers,
                'half_open_breakers': half_open_breakers,
                'breaker_states': circuit_breakers
            }
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Unable to check circuit breakers: {e}", {'error': str(e)}


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self):
        self.checkers: List[HealthChecker] = []
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_check_result: Optional[SystemHealth] = None
        
        # Register default health checkers
        self.register_checker(DatabaseHealthChecker())
        self.register_checker(SystemResourcesHealthChecker())
        self.register_checker(QuantumProvidersHealthChecker())
        self.register_checker(CircuitBreakerHealthChecker())
    
    def register_checker(self, checker: HealthChecker):
        """Register a new health checker."""
        self.checkers.append(checker)
        self.logger.info(f"Registered health checker: {checker.name}")
    
    def run_health_checks(self) -> SystemHealth:
        """Execute all health checks and return overall status."""
        self.logger.info("Running health checks...")
        
        checks = []
        for checker in self.checkers:
            try:
                check_result = checker.check()
                checks.append(check_result)
                self.logger.debug(f"Health check {checker.name}: {check_result.status.value}")
            except Exception as e:
                self.logger.error(f"Failed to execute health check {checker.name}: {e}")
                # Add a failed check result
                checks.append(HealthCheck(
                    name=checker.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check execution failed: {e}",
                    details={'error': str(e)}
                ))
        
        # Determine overall status
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            overall_status = HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            overall_status = HealthStatus.WARNING
        elif any(check.status == HealthStatus.UNKNOWN for check in checks):
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        system_health = SystemHealth(
            overall_status=overall_status,
            checks=checks
        )
        
        self._last_check_result = system_health
        self.logger.info(f"Health checks completed: {overall_status.value}")
        
        return system_health
    
    def get_last_check_result(self) -> Optional[SystemHealth]:
        """Get the last health check result without running new checks."""
        return self._last_check_result
    
    def start_monitoring(self, interval_seconds: float = 60.0):
        """Start continuous health monitoring."""
        if self._monitoring:
            self.logger.warning("Health monitoring already started")
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    self.run_health_checks()
                except Exception as e:
                    self.logger.error(f"Error in health monitoring loop: {e}")
                
                # Sleep for the specified interval
                for _ in range(int(interval_seconds * 10)):  # Check every 100ms for shutdown
                    if not self._monitoring:
                        break
                    time.sleep(0.1)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info(f"Started health monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped health monitoring")
    
    def is_healthy(self) -> bool:
        """Quick check if system is healthy based on last check."""
        if not self._last_check_result:
            # Run health checks if we don't have recent data
            self.run_health_checks()
        
        return (
            self._last_check_result and 
            self._last_check_result.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health status."""
        if not self._last_check_result:
            self.run_health_checks()
        
        if self._last_check_result:
            return self._last_check_result.to_dict()
        else:
            return {
                'overall_status': HealthStatus.UNKNOWN.value,
                'message': 'No health check data available',
                'timestamp': datetime.now().isoformat()
            }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor