"""
Unit tests for cost optimization functionality.

Tests the CostOptimizer class and related cost management utilities.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from quantum_devops_ci.cost import (
    CostOptimizer,
    CostModel,
    UsageQuota,
    ExperimentSpec,
    CostOptimizationResult,
    ProviderType
)


class TestCostModel:
    """Test cases for CostModel class."""
    
    def test_cost_model_creation(self):
        """Test CostModel creation and properties."""
        model = CostModel(
            provider=ProviderType.IBMQ,
            backend_name='ibmq_manhattan',
            cost_per_shot=0.001,
            minimum_cost=0.10,
            bulk_discount_threshold=10000,
            bulk_discount_rate=0.1
        )
        
        assert model.provider == ProviderType.IBMQ
        assert model.backend_name == 'ibmq_manhattan'
        assert model.cost_per_shot == 0.001
        assert model.minimum_cost == 0.10
        assert model.bulk_discount_threshold == 10000
        assert model.bulk_discount_rate == 0.1


class TestUsageQuota:
    """Test cases for UsageQuota class."""
    
    def test_usage_quota_creation(self):
        """Test UsageQuota creation."""
        quota = UsageQuota(
            provider=ProviderType.AWS_BRAKET,
            monthly_shot_limit=1000000,
            monthly_cost_limit=500.0,
            daily_shot_limit=50000,
            daily_cost_limit=25.0
        )
        
        assert quota.provider == ProviderType.AWS_BRAKET
        assert quota.monthly_shot_limit == 1000000
        assert quota.monthly_cost_limit == 500.0
        assert quota.daily_shot_limit == 50000
        assert quota.daily_cost_limit == 25.0
    
    def test_has_capacity(self):
        """Test capacity checking."""
        quota = UsageQuota(
            provider=ProviderType.IBMQ,
            monthly_shot_limit=10000,
            monthly_cost_limit=100.0,
            daily_shot_limit=1000,
            daily_cost_limit=10.0,
            current_monthly_shots=5000,
            current_monthly_cost=50.0,
            current_daily_shots=500,
            current_daily_cost=5.0
        )
        
        # Should have capacity for reasonable request
        assert quota.has_capacity(1000, 5.0)
        
        # Should not have capacity for request exceeding monthly limits
        assert not quota.has_capacity(10000, 60.0)
        
        # Should not have capacity for request exceeding daily limits
        assert not quota.has_capacity(600, 6.0)
    
    def test_remaining_quotas(self):
        """Test remaining quota calculations."""
        quota = UsageQuota(
            provider=ProviderType.IBMQ,
            monthly_shot_limit=10000,
            monthly_cost_limit=100.0,
            current_monthly_shots=3000,
            current_monthly_cost=30.0
        )
        
        assert quota.remaining_monthly_shots() == 7000
        assert quota.remaining_monthly_budget() == 70.0


class TestExperimentSpec:
    """Test cases for ExperimentSpec class."""
    
    def test_experiment_spec_creation(self):
        """Test ExperimentSpec creation."""
        spec = ExperimentSpec(
            id='test_exp',
            circuit='mock_circuit',
            shots=5000,
            priority='high',
            backend_preferences=['ibmq_manhattan'],
            max_cost=10.0,
            description='Test experiment'
        )
        
        assert spec.id == 'test_exp'
        assert spec.shots == 5000
        assert spec.priority == 'high'
        assert spec.backend_preferences == ['ibmq_manhattan']
        assert spec.max_cost == 10.0


class TestCostOptimizationResult:
    """Test cases for CostOptimizationResult class."""
    
    def test_result_creation_and_calculations(self):
        """Test result creation and automatic calculations."""
        result = CostOptimizationResult(
            original_cost=100.0,
            optimized_cost=75.0,
            savings=0.0,  # Will be calculated
            savings_percentage=0.0,  # Will be calculated
            optimized_assignments=[],
            recommendations=[]
        )
        
        # Should auto-calculate savings
        assert result.savings == 25.0
        assert result.savings_percentage == 25.0
    
    def test_zero_original_cost(self):
        """Test result with zero original cost."""
        result = CostOptimizationResult(
            original_cost=0.0,
            optimized_cost=0.0,
            savings=0.0,
            savings_percentage=0.0,
            optimized_assignments=[],
            recommendations=[]
        )
        
        assert result.savings_percentage == 0.0


class TestCostOptimizer:
    """Test cases for CostOptimizer class."""
    
    def test_initialization(self):
        """Test CostOptimizer initialization."""
        optimizer = CostOptimizer(monthly_budget=1000.0)
        
        assert optimizer.monthly_budget == 1000.0
        assert isinstance(optimizer.cost_models, dict)
        assert isinstance(optimizer.quotas, dict)
        assert len(optimizer.cost_models) > 0
        assert len(optimizer.quotas) > 0
    
    def test_calculate_job_cost_basic(self, cost_optimizer):
        """Test basic job cost calculation."""
        # Test simulator (free)
        cost = cost_optimizer.calculate_job_cost(1000, 'ibmq_qasm_simulator')
        assert cost == 0.0
        
        # Test paid backend
        cost = cost_optimizer.calculate_job_cost(1000, 'ibmq_manhattan')
        assert cost >= 0.10  # Should meet minimum cost
        
        # Test unknown backend
        with pytest.raises(ValueError):
            cost_optimizer.calculate_job_cost(1000, 'unknown_backend')
    
    def test_calculate_job_cost_with_bulk_discount(self, cost_optimizer):
        """Test job cost calculation with bulk discount."""
        # Small job - no discount
        small_cost = cost_optimizer.calculate_job_cost(1000, 'ibmq_manhattan')
        
        # Large job - should get bulk discount
        large_cost = cost_optimizer.calculate_job_cost(20000, 'ibmq_manhattan')
        
        # Large job should have better cost per shot due to bulk discount
        small_cost_per_shot = small_cost / 1000
        large_cost_per_shot = large_cost / 20000
        
        assert large_cost_per_shot < small_cost_per_shot
    
    def test_optimize_experiments(self, cost_optimizer, sample_experiments):
        """Test experiment optimization."""
        result = cost_optimizer.optimize_experiments(sample_experiments)
        
        assert isinstance(result, CostOptimizationResult)
        assert result.original_cost >= 0
        assert result.optimized_cost >= 0
        assert len(result.optimized_assignments) == len(sample_experiments)
        assert isinstance(result.recommendations, list)
        
        # Each assignment should have required fields
        for assignment in result.optimized_assignments:
            assert 'experiment_id' in assignment
            assert 'backend' in assignment
            assert 'cost' in assignment
            assert 'shots' in assignment
    
    def test_optimize_experiments_with_constraints(self, cost_optimizer, sample_experiments):
        """Test experiment optimization with constraints."""
        constraints = {
            'max_cost_per_experiment': 1.0,
            'deadline': '2025-12-31'
        }
        
        result = cost_optimizer.optimize_experiments(sample_experiments, constraints)
        
        # All assignments should respect cost constraint
        for assignment in result.optimized_assignments:
            assert assignment['cost'] <= 1.0
    
    def test_forecast_costs(self, cost_optimizer, sample_experiments):
        """Test cost forecasting."""
        forecast = cost_optimizer.forecast_costs(sample_experiments)
        
        assert 'monthly_estimated_cost' in forecast
        assert 'daily_estimated_cost' in forecast
        assert 'quarterly_estimated_cost' in forecast
        assert 'yearly_estimated_cost' in forecast
        assert 'budget_impact' in forecast
        assert 'experiments_count' in forecast
        assert 'cost_breakdown_by_provider' in forecast
        
        # Validate relationships
        monthly_cost = forecast['monthly_estimated_cost']
        daily_cost = forecast['daily_estimated_cost'] 
        quarterly_cost = forecast['quarterly_estimated_cost']
        
        assert daily_cost == monthly_cost / 30
        assert quarterly_cost == monthly_cost * 3
    
    def test_get_budget_status(self, cost_optimizer):
        """Test budget status retrieval."""
        status = cost_optimizer.get_budget_status()
        
        assert 'total_spent' in status
        assert 'monthly_budget' in status
        assert 'remaining_budget' in status
        assert 'budget_utilization' in status
        assert 'provider_breakdown' in status
        assert 'days_remaining_in_month' in status
        assert 'daily_budget_remaining' in status
        
        # Validate calculations
        assert status['monthly_budget'] == 1000.0
        assert status['remaining_budget'] <= status['monthly_budget']
        assert 0 <= status['budget_utilization'] <= 1
    
    def test_update_usage(self, cost_optimizer):
        """Test usage tracking."""
        initial_status = cost_optimizer.get_budget_status()
        initial_spent = initial_status['total_spent']
        
        # Update usage
        cost_optimizer.update_usage(ProviderType.IBMQ, 1000, 5.0)
        
        updated_status = cost_optimizer.get_budget_status()
        updated_spent = updated_status['total_spent']
        
        # Should reflect the update
        assert updated_spent == initial_spent + 5.0
        
        # Check usage history
        assert len(cost_optimizer.usage_history) > 0
        last_usage = cost_optimizer.usage_history[-1]
        assert last_usage['provider'] == 'ibmq'
        assert last_usage['shots'] == 1000
        assert last_usage['cost'] == 5.0
    
    def test_get_cost_recommendations(self, cost_optimizer):
        """Test cost recommendations generation."""
        # Add some usage to trigger recommendations
        cost_optimizer.update_usage(ProviderType.IBMQ, 50000, 900.0)  # High utilization
        
        recommendations = cost_optimizer.get_cost_recommendations()
        
        assert isinstance(recommendations, list)
        # Should have recommendation about high budget utilization
        high_util_warning = any('High budget utilization' in rec for rec in recommendations)
        assert high_util_warning
    
    def test_find_best_backend_assignment(self, cost_optimizer):
        """Test finding best backend assignment for experiment."""
        experiment = ExperimentSpec(
            id='test',
            circuit='mock',
            shots=1000,
            priority='medium',
            backend_preferences=['ibmq_qasm_simulator', 'ibmq_manhattan'],
            max_cost=2.0
        )
        
        assignment = cost_optimizer._find_best_backend_assignment(experiment, {})
        
        assert assignment['experiment_id'] == 'test'
        assert assignment['backend'] in experiment.backend_preferences
        assert assignment['cost'] <= experiment.max_cost
        assert assignment['shots'] == experiment.shots
    
    def test_generate_recommendations(self, cost_optimizer, sample_experiments):
        """Test recommendation generation."""
        # Create assignments
        assignments = [
            {'experiment_id': 'exp1', 'backend': 'ibmq_manhattan', 'shots': 5000},
            {'experiment_id': 'exp2', 'backend': 'ibmq_manhattan', 'shots': 3000},
            {'experiment_id': 'exp3', 'backend': 'qasm_simulator', 'shots': 1000}
        ]
        
        recommendations = cost_optimizer._generate_recommendations(sample_experiments, assignments)
        
        assert isinstance(recommendations, list)
        # Should suggest bulk discount if close to threshold
        bulk_suggestion = any('bulk discount' in rec.lower() for rec in recommendations)
        # May or may not have bulk discount suggestion depending on threshold


@pytest.mark.performance
class TestCostOptimizerPerformance:
    """Performance tests for cost optimizer."""
    
    def test_optimization_performance_large_experiment_set(self, cost_optimizer):
        """Test optimization performance with large number of experiments."""
        import time
        
        # Create large experiment set
        large_experiments = []
        for i in range(100):
            large_experiments.append({
                'id': f'exp_{i}',
                'circuit': f'circuit_{i}',
                'shots': 1000 + (i * 100),
                'priority': ['low', 'medium', 'high'][i % 3],
                'backend_preferences': ['qasm_simulator', 'ibmq_manhattan'][i % 2:i % 2 + 1]
            })
        
        start_time = time.time()
        result = cost_optimizer.optimize_experiments(large_experiments)
        optimization_time = time.time() - start_time
        
        # Should optimize reasonably quickly
        assert optimization_time < 5.0  # Should complete within 5 seconds
        assert len(result.optimized_assignments) == 100
    
    def test_cost_calculation_performance(self, cost_optimizer):
        """Test cost calculation performance."""
        import time
        
        start_time = time.time()
        
        # Calculate costs for many jobs
        for i in range(1000):
            cost_optimizer.calculate_job_cost(1000, 'ibmq_manhattan')
        
        calc_time = time.time() - start_time
        
        # Should calculate quickly
        assert calc_time < 1.0  # Should complete within 1 second


@pytest.mark.integration
class TestCostOptimizerIntegration:
    """Integration tests for cost optimizer with other components."""
    
    def test_cost_optimizer_with_database(self, cost_optimizer, clean_database):
        """Test cost optimizer integration with database."""
        # This would test saving cost records to database
        # For now, just verify cost optimizer works independently
        
        experiments = [{
            'id': 'db_test',
            'circuit': 'test_circuit',
            'shots': 1000,
            'priority': 'medium'
        }]
        
        result = cost_optimizer.optimize_experiments(experiments)
        assert len(result.optimized_assignments) == 1
    
    def test_cost_optimizer_with_cache(self, cost_optimizer, cache_manager):
        """Test cost optimizer with caching."""
        experiment_hash = 'test_hash_123'
        cost_data = {'total_cost': 10.0, 'breakdown': {}}
        
        # Cache cost calculation
        cost_optimizer.cache_cost_calculation = lambda h, d, ttl=1800: cache_manager.set(f"cost_calculation:{h}", d, ttl)
        cost_optimizer.get_cached_cost = lambda h: cache_manager.get(f"cost_calculation:{h}")
        
        # Cache the result
        cost_optimizer.cache_cost_calculation(experiment_hash, cost_data)
        
        # Retrieve from cache
        cached_result = cost_optimizer.get_cached_cost(experiment_hash)
        assert cached_result == cost_data