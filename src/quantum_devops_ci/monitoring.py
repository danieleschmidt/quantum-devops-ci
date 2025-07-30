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