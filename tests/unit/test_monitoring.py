"""
Unit tests for quantum CI monitoring functionality.

Tests the QuantumCIMonitor class and related monitoring utilities.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

from quantum_devops_ci.monitoring import (
    QuantumCIMonitor,
    MetricType,
    AlertSeverity,
    HealthStatus
)


class TestQuantumCIMonitor:
    """Test cases for QuantumCIMonitor class."""
    
    def test_initialization(self, quantum_monitor):
        """Test QuantumCIMonitor initialization."""
        assert quantum_monitor.project == "test-project"
        assert not quantum_monitor.local_storage  # Using in-memory for testing
        assert isinstance(quantum_monitor.metrics, dict)
        assert isinstance(quantum_monitor.alerts, list)
    
    def test_record_build(self, quantum_monitor, sample_build_data):
        """Test build recording functionality."""
        quantum_monitor.record_build(sample_build_data)
        
        # Check that build was recorded
        builds = quantum_monitor.get_recent_builds(1)
        assert len(builds) == 1
        assert builds[0]['commit'] == sample_build_data['commit']
        assert builds[0]['branch'] == sample_build_data['branch']
        assert builds[0]['circuit_count'] == sample_build_data['circuit_count']
    
    def test_record_hardware_usage(self, quantum_monitor, sample_hardware_usage):
        """Test hardware usage recording."""
        quantum_monitor.record_hardware_usage(sample_hardware_usage)
        
        # Check that usage was recorded
        usage_records = quantum_monitor.get_hardware_usage(1)
        assert len(usage_records) == 1
        assert usage_records[0]['backend'] == sample_hardware_usage['backend']
        assert usage_records[0]['shots'] == sample_hardware_usage['shots']
        assert usage_records[0]['cost_usd'] == sample_hardware_usage['cost_usd']
    
    def test_record_test_result(self, quantum_monitor, test_results_data):
        """Test test result recording."""
        test_result = test_results_data[0]  # Use first test result
        quantum_monitor.record_test_result(test_result)
        
        # Check that test result was recorded
        test_records = quantum_monitor.get_test_results(1)
        assert len(test_records) == 1
        assert test_records[0]['test_name'] == test_result['test_name']
        assert test_records[0]['fidelity'] == test_result['fidelity']
        assert test_records[0]['status'] == test_result['status']
    
    def test_get_build_metrics(self, quantum_monitor, sample_build_data):
        """Test build metrics calculation."""
        # Record multiple builds
        for i in range(5):
            build_data = sample_build_data.copy()
            build_data['commit'] = f'commit_{i}'
            build_data['noise_tests_passed'] = 8 if i < 4 else 6  # One build with lower success
            quantum_monitor.record_build(build_data)
        
        metrics = quantum_monitor.get_build_metrics(days=7)
        
        assert 'total_builds' in metrics
        assert 'success_rate' in metrics
        assert 'average_execution_time' in metrics
        assert 'average_fidelity' in metrics
        assert 'test_pass_rate' in metrics
        
        assert metrics['total_builds'] == 5
        assert 0 <= metrics['success_rate'] <= 1
        assert metrics['average_execution_time'] > 0
        assert 0 <= metrics['average_fidelity'] <= 1
    
    def test_get_cost_analysis(self, quantum_monitor, sample_hardware_usage):
        """Test cost analysis functionality."""
        # Record multiple usage records with different costs
        costs = [0.50, 1.25, 0.75, 2.00, 0.30]
        for i, cost in enumerate(costs):
            usage_data = sample_hardware_usage.copy()
            usage_data['cost_usd'] = cost
            usage_data['backend'] = f'backend_{i}'
            quantum_monitor.record_hardware_usage(usage_data)
        
        analysis = quantum_monitor.get_cost_analysis(days=7)
        
        assert 'total_cost' in analysis
        assert 'daily_average' in analysis
        assert 'cost_by_backend' in analysis
        assert 'cost_trend' in analysis
        
        expected_total = sum(costs)
        assert abs(analysis['total_cost'] - expected_total) < 0.01
        assert analysis['daily_average'] > 0
        assert len(analysis['cost_by_backend']) == len(costs)
    
    def test_get_performance_metrics(self, quantum_monitor, test_results_data):
        """Test performance metrics calculation."""
        # Record multiple test results
        for test_result in test_results_data:
            quantum_monitor.record_test_result(test_result)
        
        metrics = quantum_monitor.get_performance_metrics(days=7)
        
        assert 'total_tests' in metrics
        assert 'pass_rate' in metrics
        assert 'average_fidelity' in metrics
        assert 'error_rate' in metrics
        assert 'execution_time_stats' in metrics
        
        assert metrics['total_tests'] == len(test_results_data)
        assert 0 <= metrics['pass_rate'] <= 1
        assert 0 <= metrics['average_fidelity'] <= 1
        assert metrics['error_rate'] >= 0
    
    def test_get_health_status(self, quantum_monitor):
        """Test health status calculation."""
        # Record some data to have metrics to analyze
        sample_build = {
            'commit': 'health_test',
            'branch': 'main',
            'circuit_count': 3,
            'noise_tests_passed': 10,
            'noise_tests_total': 10,
            'execution_time_seconds': 25.0,
            'estimated_fidelity': 0.95
        }
        quantum_monitor.record_build(sample_build)
        
        status = quantum_monitor.get_health_status()
        
        assert 'overall_status' in status
        assert 'build_health' in status
        assert 'test_health' in status
        assert 'cost_health' in status
        assert 'alerts' in status
        assert 'last_updated' in status
        
        assert status['overall_status'] in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert isinstance(status['alerts'], list)
    
    def test_create_alert(self, quantum_monitor):
        """Test alert creation."""
        alert_data = {
            'type': 'cost_threshold',
            'severity': AlertSeverity.WARNING,
            'message': 'Daily cost exceeded threshold',
            'details': {'current_cost': 15.50, 'threshold': 10.00}
        }
        
        quantum_monitor.create_alert(alert_data)
        
        alerts = quantum_monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'cost_threshold'
        assert alerts[0]['severity'] == AlertSeverity.WARNING
        assert 'threshold' in alerts[0]['message']
    
    def test_check_cost_thresholds(self, quantum_monitor, sample_hardware_usage):
        """Test cost threshold monitoring."""
        # Set low threshold for testing
        quantum_monitor.cost_thresholds = {
            'daily_limit': 1.00,
            'monthly_limit': 20.00
        }
        
        # Record usage that exceeds daily threshold
        expensive_usage = sample_hardware_usage.copy()
        expensive_usage['cost_usd'] = 1.50
        quantum_monitor.record_hardware_usage(expensive_usage)
        
        # Check thresholds (this should create an alert)
        quantum_monitor.check_cost_thresholds()
        
        alerts = quantum_monitor.get_alerts()
        # Should have created a cost alert
        cost_alerts = [a for a in alerts if 'cost' in a.get('type', '').lower()]
        assert len(cost_alerts) > 0
    
    def test_check_performance_thresholds(self, quantum_monitor, test_results_data):
        """Test performance threshold monitoring."""
        # Set thresholds
        quantum_monitor.performance_thresholds = {
            'min_fidelity': 0.90,
            'max_error_rate': 0.05
        }
        
        # Record poor performance test
        poor_test = {
            'test_name': 'poor_performance_test',
            'framework': 'qiskit',
            'backend': 'qasm_simulator',
            'shots': 1000,
            'fidelity': 0.70,  # Below threshold
            'error_rate': 0.20,  # Above threshold
            'status': 'failed',
            'execution_time': 2.0
        }
        quantum_monitor.record_test_result(poor_test)
        
        # Check thresholds
        quantum_monitor.check_performance_thresholds()
        
        alerts = quantum_monitor.get_alerts()
        # Should have created performance alerts
        perf_alerts = [a for a in alerts if 'performance' in a.get('type', '').lower() or 'fidelity' in a.get('type', '').lower()]
        assert len(perf_alerts) > 0
    
    def test_export_metrics(self, quantum_monitor, sample_build_data, sample_hardware_usage):
        """Test metrics export functionality."""
        # Record some data
        quantum_monitor.record_build(sample_build_data)
        quantum_monitor.record_hardware_usage(sample_hardware_usage)
        
        # Export metrics
        export_data = quantum_monitor.export_metrics(days=7)
        
        assert 'project' in export_data
        assert 'export_timestamp' in export_data
        assert 'time_range_days' in export_data
        assert 'build_metrics' in export_data
        assert 'cost_analysis' in export_data
        assert 'performance_metrics' in export_data
        assert 'health_status' in export_data
        
        assert export_data['project'] == quantum_monitor.project
        assert export_data['time_range_days'] == 7
    
    def test_clear_old_data(self, quantum_monitor, sample_build_data):
        """Test old data cleanup."""
        # Record build with old timestamp
        old_build = sample_build_data.copy()
        quantum_monitor.record_build(old_build)
        
        # Manually set timestamp to be old (for testing)
        if quantum_monitor.builds:
            quantum_monitor.builds[0]['timestamp'] = datetime.now() - timedelta(days=40)
        
        # Clear data older than 30 days
        quantum_monitor.clear_old_data(days=30)
        
        # Should have no builds left
        recent_builds = quantum_monitor.get_recent_builds(40)
        assert len(recent_builds) == 0
    
    def test_get_alerts_filtering(self, quantum_monitor):
        """Test alert filtering functionality."""
        # Create alerts with different severities
        alert_configs = [
            {'type': 'test1', 'severity': AlertSeverity.INFO, 'message': 'Info alert'},
            {'type': 'test2', 'severity': AlertSeverity.WARNING, 'message': 'Warning alert'},
            {'type': 'test3', 'severity': AlertSeverity.CRITICAL, 'message': 'Critical alert'},
        ]
        
        for config in alert_configs:
            quantum_monitor.create_alert(config)
        
        # Test filtering by severity
        warning_alerts = quantum_monitor.get_alerts(severity=AlertSeverity.WARNING)
        critical_alerts = quantum_monitor.get_alerts(severity=AlertSeverity.CRITICAL)
        
        assert len(warning_alerts) == 1
        assert len(critical_alerts) == 1
        assert warning_alerts[0]['severity'] == AlertSeverity.WARNING
        assert critical_alerts[0]['severity'] == AlertSeverity.CRITICAL
    
    def test_metrics_aggregation(self, quantum_monitor, sample_build_data, sample_hardware_usage):
        """Test metrics aggregation over time periods."""
        # Record data over multiple days
        base_time = datetime.now()
        for i in range(5):
            # Builds
            build_data = sample_build_data.copy()
            build_data['commit'] = f'commit_{i}'
            quantum_monitor.record_build(build_data)
            
            # Hardware usage
            usage_data = sample_hardware_usage.copy()
            usage_data['cost_usd'] = 1.0 + (i * 0.5)
            quantum_monitor.record_hardware_usage(usage_data)
            
            # Manually adjust timestamps for testing
            if quantum_monitor.builds:
                quantum_monitor.builds[-1]['timestamp'] = base_time - timedelta(days=i)
            if quantum_monitor.hardware_usage:
                quantum_monitor.hardware_usage[-1]['timestamp'] = base_time - timedelta(days=i)
        
        # Test different time ranges
        metrics_1_day = quantum_monitor.get_build_metrics(days=1)
        metrics_7_days = quantum_monitor.get_build_metrics(days=7)
        
        # 7-day metrics should include more builds
        assert metrics_7_days['total_builds'] >= metrics_1_day['total_builds']


@pytest.mark.performance
class TestMonitoringPerformance:
    """Performance tests for monitoring functionality."""
    
    def test_large_dataset_performance(self, quantum_monitor):
        """Test monitoring performance with large datasets."""
        import time
        
        # Record large number of builds
        start_time = time.time()
        
        for i in range(100):
            build_data = {
                'commit': f'commit_{i}',
                'branch': 'main',
                'circuit_count': 3,
                'noise_tests_passed': 8,
                'noise_tests_total': 10,
                'execution_time_seconds': 20.0 + (i % 10),
                'estimated_fidelity': 0.90 + (i % 10) / 100
            }
            quantum_monitor.record_build(build_data)
        
        record_time = time.time() - start_time
        
        # Test metrics calculation performance
        start_time = time.time()
        metrics = quantum_monitor.get_build_metrics(days=30)
        metrics_time = time.time() - start_time
        
        # Should handle large datasets efficiently
        assert record_time < 2.0  # Recording should be fast
        assert metrics_time < 1.0  # Metrics calculation should be fast
        assert metrics['total_builds'] == 100
    
    def test_alert_processing_performance(self, quantum_monitor):
        """Test alert processing performance."""
        import time
        
        start_time = time.time()
        
        # Create many alerts
        for i in range(50):
            alert_data = {
                'type': f'test_alert_{i}',
                'severity': [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL][i % 3],
                'message': f'Test alert message {i}',
                'details': {'test_data': i}
            }
            quantum_monitor.create_alert(alert_data)
        
        creation_time = time.time() - start_time
        
        # Test alert retrieval performance
        start_time = time.time()
        all_alerts = quantum_monitor.get_alerts()
        warning_alerts = quantum_monitor.get_alerts(severity=AlertSeverity.WARNING)
        retrieval_time = time.time() - start_time
        
        # Should handle alerts efficiently
        assert creation_time < 1.0
        assert retrieval_time < 0.5
        assert len(all_alerts) == 50
        assert len(warning_alerts) == len([i for i in range(50) if i % 3 == 1])


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring functionality."""
    
    def test_monitoring_with_database(self, quantum_monitor, clean_database):
        """Test monitoring integration with database."""
        # This would test saving monitoring data to database
        # For now, just verify monitoring works with database available
        
        build_data = {
            'commit': 'db_integration_test',
            'branch': 'main',
            'circuit_count': 2,
            'noise_tests_passed': 5,
            'noise_tests_total': 5,
            'execution_time_seconds': 15.0,
            'estimated_fidelity': 0.93
        }
        
        quantum_monitor.record_build(build_data)
        metrics = quantum_monitor.get_build_metrics(days=1)
        
        assert metrics['total_builds'] == 1
    
    def test_monitoring_with_external_systems(self, quantum_monitor):
        """Test monitoring integration with external systems."""
        # This would test integration with external monitoring systems
        # For now, test export functionality
        
        # Record some data
        build_data = {
            'commit': 'external_test',
            'branch': 'main',
            'circuit_count': 1,
            'noise_tests_passed': 3,
            'noise_tests_total': 3,
            'execution_time_seconds': 10.0,
            'estimated_fidelity': 0.95
        }
        quantum_monitor.record_build(build_data)
        
        # Export for external systems
        export_data = quantum_monitor.export_metrics(days=1)
        
        # Should be JSON serializable (for external systems)
        json_str = json.dumps(export_data, default=str)
        parsed_data = json.loads(json_str)
        
        assert parsed_data['project'] == quantum_monitor.project
        assert 'build_metrics' in parsed_data
