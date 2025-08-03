"""
Quantum CI/CD monitoring and metrics collection.

This module provides tools for tracking quantum CI/CD pipeline metrics,
performance monitoring, and dashboard integration.
"""

import json
import time
import warnings
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from queue import Queue


@dataclass
class BuildMetrics:
    """Container for quantum build metrics."""
    commit: str
    branch: str
    timestamp: datetime
    circuit_count: int
    total_gates: int
    max_depth: int
    estimated_fidelity: float
    noise_tests_passed: int
    noise_tests_total: int
    execution_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.noise_tests_total == 0:
            return 0.0
        return self.noise_tests_passed / self.noise_tests_total


@dataclass
class HardwareUsageMetrics:
    """Container for hardware usage metrics."""
    backend: str
    provider: str
    shots: int
    queue_time_minutes: float
    execution_time_minutes: float
    cost_usd: float
    job_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    circuit_depth: Optional[int] = None
    num_qubits: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Container for performance benchmark metrics."""
    test_name: str
    framework: str
    backend: str
    execution_time_ms: float
    memory_usage_mb: float
    gate_count: int
    circuit_depth: int
    shots: int
    timestamp: datetime = field(default_factory=datetime.now)
    baseline_comparison: Optional[float] = None  # % change from baseline
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumCIMonitor:
    """
    Monitor for quantum CI/CD pipelines.
    
    This class provides comprehensive monitoring capabilities for quantum
    development workflows, including metrics collection, trend analysis,
    and dashboard integration.
    """
    
    def __init__(
        self, 
        project: str,
        dashboard_url: Optional[str] = None,
        local_storage: bool = True,
        storage_path: Optional[str] = None
    ):
        """
        Initialize quantum CI monitor.
        
        Args:
            project: Project name for metrics tracking
            dashboard_url: URL for external dashboard integration
            local_storage: Whether to store metrics locally
            storage_path: Path for local metrics storage
        """
        self.project = project
        self.dashboard_url = dashboard_url
        self.local_storage = local_storage
        
        # Set up storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / '.quantum_devops_ci' / 'metrics' / project
        
        if local_storage:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.build_metrics = []
        self.hardware_metrics = []
        self.performance_metrics = []
        
        # Background processing
        self._metric_queue = Queue()
        self._background_thread = None
        self._running = False
        
        # Load existing metrics if available
        if local_storage:
            self._load_existing_metrics()
    
    def _load_existing_metrics(self):
        """Load existing metrics from storage."""
        try:
            # Load build metrics
            build_file = self.storage_path / 'build_metrics.json'
            if build_file.exists():
                with open(build_file, 'r') as f:
                    data = json.load(f)
                    self.build_metrics = [self._deserialize_build_metric(item) for item in data]
            
            # Load hardware metrics
            hardware_file = self.storage_path / 'hardware_metrics.json'
            if hardware_file.exists():
                with open(hardware_file, 'r') as f:
                    data = json.load(f)
                    self.hardware_metrics = [self._deserialize_hardware_metric(item) for item in data]
            
            # Load performance metrics
            perf_file = self.storage_path / 'performance_metrics.json'
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    data = json.load(f)
                    self.performance_metrics = [self._deserialize_performance_metric(item) for item in data]
        
        except Exception as e:
            warnings.warn(f"Failed to load existing metrics: {e}")
    
    def _deserialize_build_metric(self, data: Dict[str, Any]) -> BuildMetrics:
        """Deserialize build metric from JSON data."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return BuildMetrics(**data)
    
    def _deserialize_hardware_metric(self, data: Dict[str, Any]) -> HardwareUsageMetrics:
        """Deserialize hardware metric from JSON data."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return HardwareUsageMetrics(**data)
    
    def _deserialize_performance_metric(self, data: Dict[str, Any]) -> PerformanceMetrics:
        """Deserialize performance metric from JSON data."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return PerformanceMetrics(**data)
    
    def start_background_processing(self):
        """Start background thread for metric processing."""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._running = True
            self._background_thread = threading.Thread(
                target=self._background_worker,
                daemon=True
            )
            self._background_thread.start()
    
    def _background_worker(self):
        """Background worker for processing metrics."""
        while self._running:
            try:
                # Process queued metrics
                if not self._metric_queue.empty():
                    metric_type, metric_data = self._metric_queue.get(timeout=1)
                    self._process_metric(metric_type, metric_data)
                else:
                    time.sleep(0.1)
            except Exception as e:
                warnings.warn(f"Background processing error: {e}")
    
    def _process_metric(self, metric_type: str, metric_data: Any):
        """Process a single metric."""
        if metric_type == 'build':
            self.build_metrics.append(metric_data)
        elif metric_type == 'hardware':
            self.hardware_metrics.append(metric_data)
        elif metric_type == 'performance':
            self.performance_metrics.append(metric_data)
        
        # Save to storage if enabled
        if self.local_storage:
            self._save_metrics()
        
        # Send to dashboard if configured
        if self.dashboard_url:
            self._send_to_dashboard(metric_type, metric_data)
    
    def record_build(self, build_data: Dict[str, Any]):
        """
        Record build metrics.
        
        Args:
            build_data: Dictionary with build information
        """
        metric = BuildMetrics(
            commit=build_data.get('commit', 'unknown'),
            branch=build_data.get('branch', 'unknown'),
            timestamp=datetime.now(),
            circuit_count=build_data.get('circuit_count', 0),
            total_gates=build_data.get('total_gates', 0),
            max_depth=build_data.get('max_depth', 0),
            estimated_fidelity=build_data.get('estimated_fidelity', 0.0),
            noise_tests_passed=build_data.get('noise_tests_passed', 0),
            noise_tests_total=build_data.get('noise_tests_total', 0),
            execution_time_seconds=build_data.get('execution_time_seconds', 0.0),
            metadata=build_data.get('metadata', {})
        )
        
        if self._background_thread and self._running:
            self._metric_queue.put(('build', metric))
        else:
            self._process_metric('build', metric)
    
    def record_hardware_usage(self, usage_data: Dict[str, Any]):
        """
        Record hardware usage metrics.
        
        Args:
            usage_data: Dictionary with hardware usage information
        """
        metric = HardwareUsageMetrics(
            backend=usage_data.get('backend', 'unknown'),
            provider=usage_data.get('provider', 'unknown'),
            shots=usage_data.get('shots', 0),
            queue_time_minutes=usage_data.get('queue_time_minutes', 0.0),
            execution_time_minutes=usage_data.get('execution_time_minutes', 0.0),
            cost_usd=usage_data.get('cost_usd', 0.0),
            job_id=usage_data.get('job_id'),
            circuit_depth=usage_data.get('circuit_depth'),
            num_qubits=usage_data.get('num_qubits'),
            success=usage_data.get('success', True),
            error_message=usage_data.get('error_message')
        )
        
        if self._background_thread and self._running:
            self._metric_queue.put(('hardware', metric))
        else:
            self._process_metric('hardware', metric)
    
    def record_performance(self, perf_data: Dict[str, Any]):
        """
        Record performance benchmark metrics.
        
        Args:
            perf_data: Dictionary with performance information
        """
        metric = PerformanceMetrics(
            test_name=perf_data.get('test_name', 'unknown'),
            framework=perf_data.get('framework', 'unknown'),
            backend=perf_data.get('backend', 'unknown'),
            execution_time_ms=perf_data.get('execution_time_ms', 0.0),
            memory_usage_mb=perf_data.get('memory_usage_mb', 0.0),
            gate_count=perf_data.get('gate_count', 0),
            circuit_depth=perf_data.get('circuit_depth', 0),
            shots=perf_data.get('shots', 0),
            baseline_comparison=perf_data.get('baseline_comparison'),
            metadata=perf_data.get('metadata', {})
        )
        
        if self._background_thread and self._running:
            self._metric_queue.put(('performance', metric))
        else:
            self._process_metric('performance', metric)
    
    def get_build_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get build metrics summary for specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with build summary statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_builds = [
            build for build in self.build_metrics
            if build.timestamp >= cutoff_date
        ]
        
        if not recent_builds:
            return {
                'total_builds': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'average_circuit_count': 0.0,
                'average_fidelity': 0.0
            }
        
        total_builds = len(recent_builds)
        successful_builds = sum(1 for build in recent_builds if build.success_rate() > 0.8)
        
        return {
            'total_builds': total_builds,
            'success_rate': successful_builds / total_builds,
            'average_execution_time': sum(b.execution_time_seconds for b in recent_builds) / total_builds,
            'average_circuit_count': sum(b.circuit_count for b in recent_builds) / total_builds,
            'average_fidelity': sum(b.estimated_fidelity for b in recent_builds) / total_builds,
            'total_circuits': sum(b.circuit_count for b in recent_builds),
            'total_gates': sum(b.total_gates for b in recent_builds),
            'max_circuit_depth': max((b.max_depth for b in recent_builds), default=0)
        }
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get cost summary for specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with cost summary statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_usage = [
            usage for usage in self.hardware_metrics
            if usage.timestamp >= cutoff_date
        ]
        
        if not recent_usage:
            return {
                'total_cost': 0.0,
                'daily_average': 0.0,
                'total_shots': 0,
                'cost_per_shot': 0.0,
                'provider_breakdown': {}
            }
        
        total_cost = sum(usage.cost_usd for usage in recent_usage)
        total_shots = sum(usage.shots for usage in recent_usage)
        
        # Provider breakdown
        provider_costs = {}
        for usage in recent_usage:
            provider = usage.provider
            if provider not in provider_costs:
                provider_costs[provider] = {'cost': 0.0, 'shots': 0}
            provider_costs[provider]['cost'] += usage.cost_usd
            provider_costs[provider]['shots'] += usage.shots
        
        return {
            'total_cost': total_cost,
            'daily_average': total_cost / days,
            'total_shots': total_shots,
            'cost_per_shot': total_cost / total_shots if total_shots > 0 else 0.0,
            'provider_breakdown': provider_costs,
            'total_jobs': len(recent_usage),
            'successful_jobs': sum(1 for usage in recent_usage if usage.success)
        }
    
    def get_performance_trends(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get performance trends for a specific test.
        
        Args:
            test_name: Name of the test to analyze
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance trend data
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        test_metrics = [
            metric for metric in self.performance_metrics
            if metric.test_name == test_name and metric.timestamp >= cutoff_date
        ]
        
        if not test_metrics:
            return {'error': f'No performance data found for test: {test_name}'}
        
        # Sort by timestamp
        test_metrics.sort(key=lambda m: m.timestamp)
        
        # Calculate trends
        execution_times = [m.execution_time_ms for m in test_metrics]
        memory_usage = [m.memory_usage_mb for m in test_metrics]
        
        return {
            'test_name': test_name,
            'data_points': len(test_metrics),
            'time_range_days': days,
            'execution_time': {
                'current': execution_times[-1] if execution_times else 0,
                'average': sum(execution_times) / len(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'trend': 'improving' if len(execution_times) > 1 and execution_times[-1] < execution_times[0] else 'stable'
            },
            'memory_usage': {
                'current': memory_usage[-1] if memory_usage else 0,
                'average': sum(memory_usage) / len(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage)
            },
            'timestamps': [m.timestamp.isoformat() for m in test_metrics]
        }
    
    def export_metrics(self, format_type: str = 'json', days: int = 30) -> str:
        """
        Export metrics to file.
        
        Args:
            format_type: Export format ('json' or 'csv')
            days: Number of days to export
            
        Returns:
            Path to exported file
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter metrics by date
        recent_builds = [b for b in self.build_metrics if b.timestamp >= cutoff_date]
        recent_hardware = [h for h in self.hardware_metrics if h.timestamp >= cutoff_date]
        recent_performance = [p for p in self.performance_metrics if p.timestamp >= cutoff_date]
        
        if format_type == 'json':
            return self._export_json(recent_builds, recent_hardware, recent_performance)
        elif format_type == 'csv':
            return self._export_csv(recent_builds, recent_hardware, recent_performance)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self, builds, hardware, performance) -> str:
        """Export metrics to JSON format."""
        export_data = {
            'project': self.project,
            'export_timestamp': datetime.now().isoformat(),
            'build_metrics': [self._serialize_build_metric(b) for b in builds],
            'hardware_metrics': [self._serialize_hardware_metric(h) for h in hardware],
            'performance_metrics': [self._serialize_performance_metric(p) for p in performance]
        }
        
        filename = f"{self.project}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)
    
    def _export_csv(self, builds, hardware, performance) -> str:
        """Export metrics to CSV format."""
        # For CSV export, we'd need pandas or manual CSV writing
        # For now, return a placeholder
        filename = f"{self.project}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w') as f:
            f.write("CSV export not yet implemented\n")
        
        return str(filepath)
    
    def _serialize_build_metric(self, metric: BuildMetrics) -> Dict[str, Any]:
        """Serialize build metric to JSON-compatible dict."""
        data = asdict(metric)
        data['timestamp'] = metric.timestamp.isoformat()
        return data
    
    def _serialize_hardware_metric(self, metric: HardwareUsageMetrics) -> Dict[str, Any]:
        """Serialize hardware metric to JSON-compatible dict."""
        data = asdict(metric)
        data['timestamp'] = metric.timestamp.isoformat()
        return data
    
    def _serialize_performance_metric(self, metric: PerformanceMetrics) -> Dict[str, Any]:
        """Serialize performance metric to JSON-compatible dict."""
        data = asdict(metric)
        data['timestamp'] = metric.timestamp.isoformat()
        return data
    
    def _save_metrics(self):
        """Save metrics to local storage."""
        try:
            # Save build metrics
            with open(self.storage_path / 'build_metrics.json', 'w') as f:
                data = [self._serialize_build_metric(m) for m in self.build_metrics]
                json.dump(data, f, indent=2)
            
            # Save hardware metrics
            with open(self.storage_path / 'hardware_metrics.json', 'w') as f:
                data = [self._serialize_hardware_metric(m) for m in self.hardware_metrics]
                json.dump(data, f, indent=2)
            
            # Save performance metrics
            with open(self.storage_path / 'performance_metrics.json', 'w') as f:
                data = [self._serialize_performance_metric(m) for m in self.performance_metrics]
                json.dump(data, f, indent=2)
        
        except Exception as e:
            warnings.warn(f"Failed to save metrics: {e}")
    
    def _send_to_dashboard(self, metric_type: str, metric_data: Any):
        """Send metric to external dashboard."""
        # This would integrate with external monitoring systems
        # For now, just log the attempt
        warnings.warn(f"Dashboard integration not yet implemented for {metric_type}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of quantum CI/CD pipeline.
        
        Returns:
            Dictionary with health metrics and status
        """
        build_summary = self.get_build_summary(7)  # Last 7 days
        cost_summary = self.get_cost_summary(7)
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct points for low build success rate
        build_success_rate = build_summary.get('success_rate', 0.0)
        if build_success_rate < 0.9:
            health_score -= (0.9 - build_success_rate) * 50
        
        # Deduct points for high cost per shot
        cost_per_shot = cost_summary.get('cost_per_shot', 0.0)
        if cost_per_shot > 0.01:  # More than 1 cent per shot
            health_score -= min(30, (cost_per_shot - 0.01) * 1000)
        
        health_score = max(0, min(100, health_score))
        
        # Determine status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 70:
            status = 'good'
        elif health_score >= 50:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'overall_status': status,
            'health_score': health_score,
            'build_success_rate': build_success_rate,
            'recent_builds': build_summary.get('total_builds', 0),
            'cost_per_shot': cost_per_shot,
            'total_weekly_cost': cost_summary.get('total_cost', 0.0),
            'recommendations': self._get_health_recommendations(build_summary, cost_summary)
        }
    
    def _get_health_recommendations(self, build_summary: Dict, cost_summary: Dict) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if build_summary.get('success_rate', 0.0) < 0.8:
            recommendations.append("⚠️ Low build success rate. Review test failures and noise model configuration.")
        
        if cost_summary.get('cost_per_shot', 0.0) > 0.005:
            recommendations.append("💰 High cost per shot. Consider using simulators for development and optimize job batching.")
        
        if build_summary.get('total_builds', 0) == 0:
            recommendations.append("📊 No recent builds detected. Ensure CI/CD pipeline is properly configured.")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown monitoring and save final metrics."""
        self._running = False
        
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5)
        
        if self.local_storage:
            self._save_metrics()
        self.local_storage = local_storage
        
        # Set up storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.cwd() / "quantum_metrics"
        
        if self.local_storage:
            self.storage_path.mkdir(exist_ok=True)
        
        # Metrics storage
        self.build_metrics: List[BuildMetrics] = []
        self.hardware_metrics: List[HardwareUsageMetrics] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Background processing
        self.metrics_queue = Queue()
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Load existing metrics if available
        self._load_existing_metrics()
        
        # Start background processing
        self._start_background_processing()
    
    def record_build(self, build_data: Dict[str, Any]) -> str:
        """
        Record quantum build metrics.
        
        Args:
            build_data: Dictionary containing build information
            
        Returns:
            Metrics record ID
        """
        metrics = BuildMetrics(
            commit=build_data["commit"],
            branch=build_data.get("branch", "unknown"),
            timestamp=datetime.now(),
            circuit_count=build_data["circuit_count"],
            total_gates=build_data["total_gates"],
            max_depth=build_data["max_depth"],
            estimated_fidelity=build_data["estimated_fidelity"],
            noise_tests_passed=build_data["noise_tests_passed"],
            noise_tests_total=build_data["noise_tests_total"],
            execution_time_seconds=build_data.get("execution_time_seconds", 0.0),
            metadata=build_data.get("metadata", {})
        )
        
        self.build_metrics.append(metrics)
        
        # Queue for background processing
        self.metrics_queue.put(("build", metrics))
        
        record_id = f"build_{metrics.commit}_{int(metrics.timestamp.timestamp())}"
        return record_id
    
    def record_hardware_usage(self, usage_data: Dict[str, Any]) -> str:
        """
        Record quantum hardware usage metrics.
        
        Args:
            usage_data: Dictionary containing hardware usage information
            
        Returns:
            Metrics record ID
        """
        metrics = HardwareUsageMetrics(
            backend=usage_data["backend"],
            provider=usage_data.get("provider", "unknown"),
            shots=usage_data["shots"],
            queue_time_minutes=usage_data["queue_time_minutes"],
            execution_time_minutes=usage_data["execution_time_minutes"],
            cost_usd=usage_data["cost_usd"],
            job_id=usage_data.get("job_id"),
            circuit_depth=usage_data.get("circuit_depth"),
            num_qubits=usage_data.get("num_qubits"),
            success=usage_data.get("success", True),
            error_message=usage_data.get("error_message")
        )
        
        self.hardware_metrics.append(metrics)
        
        # Queue for background processing
        self.metrics_queue.put(("hardware", metrics))
        
        record_id = f"hw_{metrics.backend}_{int(metrics.timestamp.timestamp())}"
        return record_id
    
    def record_performance(self, performance_data: Dict[str, Any]) -> str:
        """
        Record performance benchmark metrics.
        
        Args:
            performance_data: Dictionary containing performance information
            
        Returns:
            Metrics record ID
        """
        metrics = PerformanceMetrics(
            test_name=performance_data["test_name"],
            framework=performance_data["framework"],
            backend=performance_data["backend"],
            execution_time_ms=performance_data["execution_time_ms"],
            memory_usage_mb=performance_data["memory_usage_mb"],
            gate_count=performance_data["gate_count"],
            circuit_depth=performance_data["circuit_depth"],
            shots=performance_data["shots"],
            baseline_comparison=performance_data.get("baseline_comparison"),
            metadata=performance_data.get("metadata", {})
        )
        
        self.performance_metrics.append(metrics)
        
        # Queue for background processing
        self.metrics_queue.put(("performance", metrics))
        
        record_id = f"perf_{metrics.test_name}_{int(metrics.timestamp.timestamp())}"
        return record_id
    
    def get_build_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get build metrics summary for recent period.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Summary statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_builds = [b for b in self.build_metrics if b.timestamp >= cutoff_date]
        
        if not recent_builds:
            return {"period_days": days, "total_builds": 0}
        
        return {
            "period_days": days,
            "total_builds": len(recent_builds),
            "success_rate": sum(b.success_rate() for b in recent_builds) / len(recent_builds),
            "avg_circuit_count": sum(b.circuit_count for b in recent_builds) / len(recent_builds),
            "avg_gate_count": sum(b.total_gates for b in recent_builds) / len(recent_builds),
            "max_depth": max(b.max_depth for b in recent_builds),
            "avg_fidelity": sum(b.estimated_fidelity for b in recent_builds) / len(recent_builds),
            "total_execution_time": sum(b.execution_time_seconds for b in recent_builds),
            "builds_per_day": len(recent_builds) / days
        }
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get cost summary for recent period.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Cost analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_usage = [h for h in self.hardware_metrics if h.timestamp >= cutoff_date]
        
        if not recent_usage:
            return {"period_days": days, "total_cost": 0.0}
        
        total_cost = sum(h.cost_usd for h in recent_usage)
        cost_by_provider = {}
        cost_by_backend = {}
        
        for usage in recent_usage:
            # By provider
            provider_cost = cost_by_provider.get(usage.provider, 0.0)
            cost_by_provider[usage.provider] = provider_cost + usage.cost_usd
            
            # By backend
            backend_cost = cost_by_backend.get(usage.backend, 0.0)
            cost_by_backend[usage.backend] = backend_cost + usage.cost_usd
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "daily_average": total_cost / days,
            "cost_by_provider": cost_by_provider,
            "cost_by_backend": cost_by_backend,
            "total_shots": sum(h.shots for h in recent_usage),
            "avg_cost_per_shot": total_cost / sum(h.shots for h in recent_usage) if recent_usage else 0.0
        }
    
    def get_performance_trends(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get performance trends for specific test.
        
        Args:
            test_name: Name of performance test
            days: Number of days to analyze
            
        Returns:
            Performance trend analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        test_metrics = [p for p in self.performance_metrics 
                       if p.test_name == test_name and p.timestamp >= cutoff_date]
        
        if not test_metrics:
            return {"test_name": test_name, "data_points": 0}
        
        # Sort by timestamp
        test_metrics.sort(key=lambda x: x.timestamp)
        
        execution_times = [m.execution_time_ms for m in test_metrics]
        memory_usage = [m.memory_usage_mb for m in test_metrics]
        
        return {
            "test_name": test_name,
            "data_points": len(test_metrics),
            "execution_time": {
                "current": execution_times[-1],
                "average": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "trend": self._calculate_trend(execution_times)
            },
            "memory_usage": {
                "current": memory_usage[-1],
                "average": sum(memory_usage) / len(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage),
                "trend": self._calculate_trend(memory_usage)
            },
            "latest_baseline_comparison": test_metrics[-1].baseline_comparison
        }
    
    def export_metrics(self, format: str = "json", file_path: Optional[str] = None) -> str:
        """
        Export all metrics to file.
        
        Args:
            format: Export format ("json", "csv")
            file_path: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.storage_path / f"quantum_metrics_{timestamp}.{format}")
        
        if format == "json":
            self._export_json(file_path)
        elif format == "csv":
            self._export_csv(file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return file_path
    
    def send_to_dashboard(self, metrics_type: str = "all") -> bool:
        """
        Send metrics to external dashboard.
        
        Args:
            metrics_type: Type of metrics to send ("all", "build", "hardware", "performance")
            
        Returns:
            True if successful
        """
        if not self.dashboard_url:
            warnings.warn("No dashboard URL configured")
            return False
        
        try:
            # Prepare metrics data
            data = {"project": self.project, "timestamp": datetime.now().isoformat()}
            
            if metrics_type in ["all", "build"]:
                data["build_metrics"] = [asdict(m) for m in self.build_metrics[-100:]]  # Last 100
            
            if metrics_type in ["all", "hardware"]:
                data["hardware_metrics"] = [asdict(m) for m in self.hardware_metrics[-100:]]
            
            if metrics_type in ["all", "performance"]:
                data["performance_metrics"] = [asdict(m) for m in self.performance_metrics[-100:]]
            
            # Send to dashboard (placeholder implementation)
            warnings.warn("Dashboard integration is not yet implemented")
            # In real implementation, this would send HTTP request to dashboard API
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to send metrics to dashboard: {e}")
            return False
    
    def cleanup_old_metrics(self, days: int = 90):
        """
        Clean up metrics older than specified days.
        
        Args:
            days: Age threshold for cleanup
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean up in-memory metrics
        self.build_metrics = [m for m in self.build_metrics if m.timestamp >= cutoff_date]
        self.hardware_metrics = [m for m in self.hardware_metrics if m.timestamp >= cutoff_date]
        self.performance_metrics = [m for m in self.performance_metrics if m.timestamp >= cutoff_date]
        
        # Save cleaned metrics
        if self.local_storage:
            self._save_metrics()
    
    def shutdown(self):
        """Shutdown monitor and save pending metrics."""
        # Stop background processing
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        # Save any pending metrics
        if self.local_storage:
            self._save_metrics()
    
    def _start_background_processing(self):
        """Start background thread for metrics processing."""
        def process_metrics():
            while not self.stop_processing.is_set():
                try:
                    # Process queued metrics
                    while not self.metrics_queue.empty():
                        metrics_type, metrics = self.metrics_queue.get_nowait()
                        self._process_metrics(metrics_type, metrics)
                    
                    # Periodic save
                    if self.local_storage:
                        self._save_metrics()
                    
                    time.sleep(10)  # Process every 10 seconds
                    
                except Exception as e:
                    warnings.warn(f"Error in metrics processing: {e}")
        
        self.processing_thread = threading.Thread(target=process_metrics, daemon=True)
        self.processing_thread.start()
    
    def _process_metrics(self, metrics_type: str, metrics: Any):
        """Process individual metrics record."""
        # Add any additional processing logic here
        # For example: alerting, aggregation, external API calls
        pass
    
    def _load_existing_metrics(self):
        """Load existing metrics from storage."""
        if not self.local_storage:
            return
        
        try:
            # Load build metrics
            build_file = self.storage_path / "build_metrics.json"
            if build_file.exists():
                with open(build_file, 'r') as f:
                    data = json.load(f)
                    self.build_metrics = [
                        BuildMetrics(**item) for item in data
                    ]
            
            # Load hardware metrics
            hardware_file = self.storage_path / "hardware_metrics.json"
            if hardware_file.exists():
                with open(hardware_file, 'r') as f:
                    data = json.load(f)
                    self.hardware_metrics = [
                        HardwareUsageMetrics(**item) for item in data
                    ]
            
            # Load performance metrics
            perf_file = self.storage_path / "performance_metrics.json"
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    data = json.load(f)
                    self.performance_metrics = [
                        PerformanceMetrics(**item) for item in data
                    ]
                    
        except Exception as e:
            warnings.warn(f"Failed to load existing metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to local storage."""
        if not self.local_storage:
            return
        
        try:
            # Save build metrics
            build_file = self.storage_path / "build_metrics.json"
            with open(build_file, 'w') as f:
                data = [asdict(m) for m in self.build_metrics]
                # Convert datetime objects to ISO strings
                for item in data:
                    item['timestamp'] = item['timestamp'].isoformat()
                json.dump(data, f, indent=2)
            
            # Save hardware metrics
            hardware_file = self.storage_path / "hardware_metrics.json"
            with open(hardware_file, 'w') as f:
                data = [asdict(m) for m in self.hardware_metrics]
                for item in data:
                    item['timestamp'] = item['timestamp'].isoformat()
                json.dump(data, f, indent=2)
            
            # Save performance metrics
            perf_file = self.storage_path / "performance_metrics.json"
            with open(perf_file, 'w') as f:
                data = [asdict(m) for m in self.performance_metrics]
                for item in data:
                    item['timestamp'] = item['timestamp'].isoformat()
                json.dump(data, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Failed to save metrics: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if abs(slope) < 0.01:  # Threshold for "stable"
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _export_json(self, file_path: str):
        """Export metrics to JSON format."""
        data = {
            "project": self.project,
            "export_timestamp": datetime.now().isoformat(),
            "build_metrics": [asdict(m) for m in self.build_metrics],
            "hardware_metrics": [asdict(m) for m in self.hardware_metrics],
            "performance_metrics": [asdict(m) for m in self.performance_metrics]
        }
        
        # Convert datetime objects to ISO strings
        for metrics_list in [data["build_metrics"], data["hardware_metrics"], data["performance_metrics"]]:
            for item in metrics_list:
                if 'timestamp' in item:
                    item['timestamp'] = item['timestamp'].isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, file_path: str):
        """Export metrics to CSV format."""
        # Placeholder for CSV export
        warnings.warn("CSV export not yet implemented")


def main():
    """Main entry point for quantum monitoring CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum CI/CD monitoring')
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--dashboard', help='Dashboard URL')
    parser.add_argument('--export', choices=['json', 'csv'], help='Export format')
    parser.add_argument('--summary', action='store_true', help='Show metrics summary')
    parser.add_argument('--days', type=int, default=30, help='Days to include in analysis')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = QuantumCIMonitor(args.project, args.dashboard)
    
    try:
        if args.summary:
            # Show summary
            build_summary = monitor.get_build_summary(args.days)
            cost_summary = monitor.get_cost_summary(args.days)
            
            print(f"Quantum CI/CD Metrics Summary ({args.days} days)")
            print("=" * 50)
            print(f"Project: {args.project}")
            print(f"Total builds: {build_summary['total_builds']}")
            print(f"Success rate: {build_summary.get('success_rate', 0.0):.1%}")
            print(f"Total cost: ${cost_summary['total_cost']:.2f}")
            print(f"Daily average cost: ${cost_summary['daily_average']:.2f}")
        
        if args.export:
            # Export metrics
            file_path = monitor.export_metrics(args.export)
            print(f"Metrics exported to: {file_path}")
    
    finally:
        monitor.shutdown()


if __name__ == '__main__':
    main()