"""
Unit tests for quantum job scheduling functionality.

Tests the QuantumJobScheduler class and related scheduling utilities.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from quantum_devops_ci.scheduling import (
    QuantumJobScheduler,
    BackendInfo,
    OptimizedSchedule,
    JobPriority,
    ResourceConstraints
)


class TestBackendInfo:
    """Test cases for BackendInfo class."""
    
    def test_backend_info_creation(self):
        """Test BackendInfo creation and properties."""
        backend = BackendInfo(
            name='ibmq_manhattan',
            provider='ibmq',
            qubits=65,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
            coupling_map=[[0, 1], [1, 2]],
            queue_length=15,
            estimated_wait_minutes=45,
            cost_per_shot=0.001,
            noise_level=0.02,
            availability=0.95
        )
        
        assert backend.name == 'ibmq_manhattan'
        assert backend.provider == 'ibmq'
        assert backend.qubits == 65
        assert 'cx' in backend.basis_gates
        assert backend.queue_length == 15
        assert backend.estimated_wait_minutes == 45
        assert backend.cost_per_shot == 0.001
        assert backend.noise_level == 0.02
        assert backend.availability == 0.95


class TestOptimizedSchedule:
    """Test cases for OptimizedSchedule class."""
    
    def test_optimized_schedule_creation(self):
        """Test OptimizedSchedule creation."""
        assignments = [
            {'job_id': 'job1', 'backend': 'ibmq_manhattan', 'estimated_start': datetime.now()},
            {'job_id': 'job2', 'backend': 'qasm_simulator', 'estimated_start': datetime.now()}
        ]
        
        schedule = OptimizedSchedule(
            assignments=assignments,
            total_estimated_cost=15.50,
            total_estimated_time_minutes=120,
            optimization_score=0.85,
            recommendations=['Use simulator for development tests']
        )
        
        assert len(schedule.assignments) == 2
        assert schedule.total_estimated_cost == 15.50
        assert schedule.total_estimated_time_minutes == 120
        assert schedule.optimization_score == 0.85
        assert len(schedule.recommendations) == 1


class TestResourceConstraints:
    """Test cases for ResourceConstraints class."""
    
    def test_resource_constraints_creation(self):
        """Test ResourceConstraints creation."""
        constraints = ResourceConstraints(
            max_cost=100.0,
            max_time_minutes=180,
            preferred_providers=['ibmq', 'aws_braket'],
            deadline=datetime.now() + timedelta(hours=24),
            priority_weights={'high': 3.0, 'medium': 2.0, 'low': 1.0}
        )
        
        assert constraints.max_cost == 100.0
        assert constraints.max_time_minutes == 180
        assert 'ibmq' in constraints.preferred_providers
        assert constraints.priority_weights['high'] == 3.0


class TestQuantumJobScheduler:
    """Test cases for QuantumJobScheduler class."""
    
    def test_initialization(self, job_scheduler):
        """Test QuantumJobScheduler initialization."""
        assert job_scheduler.optimization_goal == "minimize_cost"
        assert isinstance(job_scheduler.backends, dict)
        assert len(job_scheduler.backends) > 0
    
    def test_get_available_backends(self, job_scheduler):
        """Test getting available backends."""
        backends = job_scheduler.get_available_backends()
        
        assert isinstance(backends, list)
        assert len(backends) > 0
        
        # Should include simulators
        backend_names = [b.name for b in backends]
        assert 'qasm_simulator' in backend_names
    
    def test_estimate_job_metrics(self, job_scheduler):
        """Test job metrics estimation."""
        job = {
            'id': 'test_job',
            'circuit': 'mock_circuit',
            'shots': 1000,
            'priority': JobPriority.MEDIUM,
            'backend_preferences': ['ibmq_manhattan', 'qasm_simulator']
        }
        
        metrics = job_scheduler.estimate_job_metrics(job)
        
        assert 'estimated_cost' in metrics
        assert 'estimated_time_minutes' in metrics
        assert 'complexity_score' in metrics
        assert 'resource_requirements' in metrics
        assert metrics['estimated_cost'] >= 0
        assert metrics['estimated_time_minutes'] >= 0
    
    def test_optimize_schedule_basic(self, job_scheduler, sample_jobs):
        """Test basic schedule optimization."""
        schedule = job_scheduler.optimize_schedule(sample_jobs)
        
        assert isinstance(schedule, OptimizedSchedule)
        assert len(schedule.assignments) == len(sample_jobs)
        assert schedule.total_estimated_cost >= 0
        assert schedule.total_estimated_time_minutes >= 0
        assert 0 <= schedule.optimization_score <= 1
        
        # Each assignment should have required fields
        for assignment in schedule.assignments:
            assert 'job_id' in assignment
            assert 'backend' in assignment
            assert 'estimated_start' in assignment
            assert 'estimated_cost' in assignment
    
    def test_optimize_schedule_with_constraints(self, job_scheduler, sample_jobs):
        """Test schedule optimization with constraints."""
        constraints = ResourceConstraints(
            max_cost=10.0,
            max_time_minutes=60,
            preferred_providers=['ibmq']
        )
        
        schedule = job_scheduler.optimize_schedule(sample_jobs, constraints)
        
        # Should respect cost constraint
        assert schedule.total_estimated_cost <= constraints.max_cost
        
        # Should prefer specified providers
        for assignment in schedule.assignments:
            backend_name = assignment['backend']
            # Check if backend belongs to preferred provider
            # (This is a simplified check)
            if backend_name not in ['qasm_simulator', 'statevector_simulator']:
                assert any(provider in backend_name for provider in constraints.preferred_providers)
    
    def test_get_backend_recommendation(self, job_scheduler):
        """Test backend recommendation for a job."""
        job = {
            'id': 'recommendation_test',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': JobPriority.HIGH,
            'max_cost': 5.0
        }
        
        recommendation = job_scheduler.get_backend_recommendation(job)
        
        assert 'recommended_backend' in recommendation
        assert 'estimated_cost' in recommendation
        assert 'estimated_wait_time' in recommendation
        assert 'confidence' in recommendation
        assert 'alternatives' in recommendation
        
        # Cost should be within job's budget
        assert recommendation['estimated_cost'] <= job['max_cost']
        assert 0 <= recommendation['confidence'] <= 1
    
    def test_update_backend_status(self, job_scheduler):
        """Test updating backend status."""
        backend_name = 'test_backend'
        status_update = {
            'queue_length': 10,
            'estimated_wait_minutes': 30,
            'availability': 0.98,
            'last_updated': datetime.now()
        }
        
        job_scheduler.update_backend_status(backend_name, status_update)
        
        # Should update backend info if it exists
        if backend_name in job_scheduler.backends:
            backend = job_scheduler.backends[backend_name]
            assert backend.queue_length == 10
            assert backend.estimated_wait_minutes == 30
            assert backend.availability == 0.98
    
    def test_get_job_dependencies(self, job_scheduler):
        """Test job dependency analysis."""
        jobs = [
            {'id': 'job1', 'depends_on': []},
            {'id': 'job2', 'depends_on': ['job1']},
            {'id': 'job3', 'depends_on': ['job1', 'job2']}
        ]
        
        dependencies = job_scheduler.get_job_dependencies(jobs)
        
        assert 'job1' in dependencies
        assert 'job2' in dependencies
        assert 'job3' in dependencies
        
        # job1 should have no dependencies
        assert len(dependencies['job1']) == 0
        
        # job2 should depend on job1
        assert 'job1' in dependencies['job2']
        
        # job3 should depend on job1 and job2
        assert 'job1' in dependencies['job3']
        assert 'job2' in dependencies['job3']
    
    def test_calculate_priority_score(self, job_scheduler):
        """Test priority score calculation."""
        high_priority_job = {'priority': JobPriority.CRITICAL, 'deadline': datetime.now() + timedelta(hours=1)}
        medium_priority_job = {'priority': JobPriority.MEDIUM, 'deadline': datetime.now() + timedelta(days=1)}
        low_priority_job = {'priority': JobPriority.LOW, 'deadline': datetime.now() + timedelta(days=7)}
        
        high_score = job_scheduler._calculate_priority_score(high_priority_job)
        medium_score = job_scheduler._calculate_priority_score(medium_priority_job)
        low_score = job_scheduler._calculate_priority_score(low_priority_job)
        
        # Higher priority should have higher score
        assert high_score > medium_score > low_score
        assert all(score >= 0 for score in [high_score, medium_score, low_score])
    
    def test_find_optimal_assignment(self, job_scheduler):
        """Test finding optimal job-backend assignment."""
        job = {
            'id': 'optimal_test',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': JobPriority.MEDIUM,
            'backend_preferences': ['qasm_simulator', 'ibmq_manhattan']
        }
        
        assignment = job_scheduler._find_optimal_assignment(job, {})
        
        assert assignment['job_id'] == 'optimal_test'
        assert assignment['backend'] in job['backend_preferences']
        assert 'estimated_cost' in assignment
        assert 'estimated_start' in assignment
        assert assignment['estimated_cost'] >= 0


@pytest.fixture
def sample_jobs():
    """Sample jobs for testing."""
    return [
        {
            'id': 'vqe_job',
            'circuit': 'vqe_circuit',
            'shots': 5000,
            'priority': JobPriority.HIGH,
            'backend_preferences': ['ibmq_manhattan'],
            'max_cost': 10.0
        },
        {
            'id': 'qaoa_job',
            'circuit': 'qaoa_circuit',
            'shots': 2000,
            'priority': JobPriority.MEDIUM,
            'backend_preferences': ['aws_sv1'],
            'max_cost': 5.0
        },
        {
            'id': 'test_job',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': JobPriority.LOW,
            'backend_preferences': ['qasm_simulator'],
            'max_cost': 0.0
        }
    ]


@pytest.mark.performance
class TestSchedulingPerformance:
    """Performance tests for job scheduling."""
    
    def test_large_job_set_optimization(self, job_scheduler):
        """Test optimization performance with large job set."""
        import time
        
        # Create large job set
        large_jobs = []
        for i in range(50):
            large_jobs.append({
                'id': f'job_{i}',
                'circuit': f'circuit_{i}',
                'shots': 1000 + (i * 100),
                'priority': [JobPriority.LOW, JobPriority.MEDIUM, JobPriority.HIGH][i % 3],
                'backend_preferences': ['qasm_simulator', 'ibmq_manhattan'][i % 2:i % 2 + 1]
            })
        
        start_time = time.time()
        schedule = job_scheduler.optimize_schedule(large_jobs)
        optimization_time = time.time() - start_time
        
        # Should optimize reasonably quickly
        assert optimization_time < 3.0  # Should complete within 3 seconds
        assert len(schedule.assignments) == 50
    
    def test_backend_recommendation_performance(self, job_scheduler):
        """Test backend recommendation performance."""
        import time
        
        job = {
            'id': 'perf_test',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': JobPriority.MEDIUM
        }
        
        start_time = time.time()
        
        # Generate many recommendations
        for i in range(100):
            job['id'] = f'perf_test_{i}'
            recommendation = job_scheduler.get_backend_recommendation(job)
            assert 'recommended_backend' in recommendation
        
        total_time = time.time() - start_time
        
        # Should generate recommendations quickly
        assert total_time < 1.0  # Should complete within 1 second


@pytest.mark.integration
class TestSchedulingIntegration:
    """Integration tests for job scheduling."""
    
    def test_scheduler_with_cost_optimizer(self, job_scheduler, cost_optimizer):
        """Test scheduler integration with cost optimizer."""
        jobs = [{
            'id': 'integration_test',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': JobPriority.MEDIUM
        }]
        
        # Get schedule from scheduler
        schedule = job_scheduler.optimize_schedule(jobs)
        
        # Extract experiments for cost optimizer
        experiments = []
        for assignment in schedule.assignments:
            experiments.append({
                'id': assignment['job_id'],
                'circuit': 'test_circuit',
                'shots': 1000,
                'backend_preferences': [assignment['backend']]
            })
        
        # Optimize costs
        cost_result = cost_optimizer.optimize_experiments(experiments)
        
        assert len(cost_result.optimized_assignments) == len(experiments)
    
    def test_scheduler_with_database(self, job_scheduler, clean_database):
        """Test scheduler with database integration."""
        # This would test saving job records to database
        # For now, just verify scheduler works independently
        
        jobs = [{
            'id': 'db_integration_test',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': JobPriority.MEDIUM
        }]
        
        schedule = job_scheduler.optimize_schedule(jobs)
        assert len(schedule.assignments) == 1
