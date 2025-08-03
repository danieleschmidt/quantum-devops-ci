"""
Unit tests for quantum deployment functionality.

Tests the quantum deployment classes and A/B testing utilities.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import statistics

from quantum_devops_ci.deployment import (
    QuantumDeployment,
    QuantumABTest,
    DeploymentStrategy,
    ABTestResult,
    ABTestAnalysis,
    DeploymentConfig
)


class TestDeploymentConfig:
    """Test cases for DeploymentConfig class."""
    
    def test_config_creation(self):
        """Test DeploymentConfig creation."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.BLUE_GREEN,
            rollback_threshold=0.05,
            health_check_timeout=300,
            traffic_split={'blue': 0.8, 'green': 0.2},
            monitoring_period_minutes=60
        )
        
        assert config.strategy == DeploymentStrategy.BLUE_GREEN
        assert config.rollback_threshold == 0.05
        assert config.health_check_timeout == 300
        assert config.traffic_split['blue'] == 0.8
        assert config.monitoring_period_minutes == 60


class TestABTestResult:
    """Test cases for ABTestResult class."""
    
    def test_result_creation(self):
        """Test ABTestResult creation and calculations."""
        measurements = [0.95, 0.92, 0.94, 0.96, 0.93]
        
        result = ABTestResult(
            algorithm_name='VQE_v2',
            measurements=measurements,
            sample_size=len(measurements),
            execution_time_seconds=120.5,
            cost_usd=5.75,
            success_rate=0.96
        )
        
        assert result.algorithm_name == 'VQE_v2'
        assert len(result.measurements) == 5
        assert result.sample_size == 5
        assert result.execution_time_seconds == 120.5
        assert result.cost_usd == 5.75
        assert result.success_rate == 0.96
        
        # Test calculated properties
        assert abs(result.mean - statistics.mean(measurements)) < 0.001
        assert abs(result.std_dev - statistics.stdev(measurements)) < 0.001
        assert result.confidence_interval[0] < result.mean < result.confidence_interval[1]


class TestABTestAnalysis:
    """Test cases for ABTestAnalysis class."""
    
    def test_analysis_creation(self):
        """Test ABTestAnalysis creation."""
        analysis = ABTestAnalysis(
            winner='algorithm_a',
            confidence_level=0.95,
            p_value=0.02,
            effect_size=0.15,
            recommendation='Deploy algorithm_a',
            statistical_significance=True,
            practical_significance=True
        )
        
        assert analysis.winner == 'algorithm_a'
        assert analysis.confidence_level == 0.95
        assert analysis.p_value == 0.02
        assert analysis.effect_size == 0.15
        assert 'Deploy' in analysis.recommendation
        assert analysis.statistical_significance
        assert analysis.practical_significance


class TestQuantumDeployment:
    """Test cases for QuantumDeployment class."""
    
    def test_initialization(self):
        """Test QuantumDeployment initialization."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            rollback_threshold=0.1
        )
        
        deployment = QuantumDeployment("test-service", config)
        
        assert deployment.service_name == "test-service"
        assert deployment.config.strategy == DeploymentStrategy.CANARY
        assert deployment.current_deployment is None
        assert isinstance(deployment.deployment_history, list)
    
    def test_deploy_blue_green(self):
        """Test blue-green deployment strategy."""
        config = DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN)
        deployment = QuantumDeployment("quantum-service", config)
        
        # Mock algorithm/service to deploy
        algorithm_v2 = {
            'name': 'VQE_v2.0',
            'version': '2.0.0',
            'circuit': 'mock_circuit_v2',
            'parameters': {'layers': 4, 'optimizer': 'COBYLA'}
        }
        
        result = deployment.deploy_blue_green(algorithm_v2)
        
        assert result['strategy'] == 'blue_green'
        assert result['status'] in ['success', 'failed', 'pending']
        assert 'deployment_id' in result
        assert 'timestamp' in result
        
        # Should have current deployment set
        assert deployment.current_deployment is not None
        assert deployment.current_deployment['version'] == '2.0.0'
    
    def test_deploy_canary(self):
        """Test canary deployment strategy."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            traffic_split={'current': 0.9, 'canary': 0.1}
        )
        deployment = QuantumDeployment("quantum-service", config)
        
        algorithm_canary = {
            'name': 'QAOA_canary',
            'version': '1.5.0-canary',
            'circuit': 'mock_qaoa_canary',
            'parameters': {'p': 3, 'mixer': 'XY'}
        }
        
        result = deployment.deploy_canary(algorithm_canary, traffic_percentage=10)
        
        assert result['strategy'] == 'canary'
        assert result['traffic_percentage'] == 10
        assert 'deployment_id' in result
        assert 'health_checks' in result
    
    def test_deploy_rolling(self):
        """Test rolling deployment strategy."""
        config = DeploymentConfig(strategy=DeploymentStrategy.ROLLING)
        deployment = QuantumDeployment("quantum-service", config)
        
        algorithm_rolling = {
            'name': 'VQE_rolling',
            'version': '1.2.0',
            'circuit': 'mock_vqe_rolling'
        }
        
        result = deployment.deploy_rolling(algorithm_rolling, batch_size=2)
        
        assert result['strategy'] == 'rolling'
        assert result['batch_size'] == 2
        assert 'rollout_plan' in result
        assert isinstance(result['rollout_plan'], list)
    
    def test_health_check(self):
        """Test deployment health checking."""
        config = DeploymentConfig()
        deployment = QuantumDeployment("quantum-service", config)
        
        # Set up mock current deployment
        deployment.current_deployment = {
            'name': 'VQE_v1',
            'version': '1.0.0',
            'deployment_id': 'test-deployment-123'
        }
        
        health_status = deployment.check_health()
        
        assert 'status' in health_status
        assert 'checks' in health_status
        assert 'timestamp' in health_status
        assert health_status['status'] in ['healthy', 'unhealthy', 'degraded']
        
        # Should include basic health checks
        check_names = [check['name'] for check in health_status['checks']]
        assert 'deployment_exists' in check_names
    
    def test_rollback(self):
        """Test deployment rollback functionality."""
        config = DeploymentConfig()
        deployment = QuantumDeployment("quantum-service", config)
        
        # Set up deployment history
        deployment.deployment_history = [
            {
                'deployment_id': 'deploy-1',
                'version': '1.0.0',
                'timestamp': datetime.now() - timedelta(hours=2),
                'status': 'success'
            },
            {
                'deployment_id': 'deploy-2',
                'version': '2.0.0',
                'timestamp': datetime.now() - timedelta(hours=1),
                'status': 'failed'
            }
        ]
        
        rollback_result = deployment.rollback(reason="Performance degradation")
        
        assert rollback_result['action'] == 'rollback'
        assert 'previous_version' in rollback_result
        assert 'reason' in rollback_result
        assert rollback_result['reason'] == "Performance degradation"
    
    def test_get_deployment_metrics(self):
        """Test deployment metrics collection."""
        config = DeploymentConfig()
        deployment = QuantumDeployment("quantum-service", config)
        
        # Add some deployment history
        deployment.deployment_history = [
            {'deployment_id': 'deploy-1', 'status': 'success', 'execution_time': 120},
            {'deployment_id': 'deploy-2', 'status': 'success', 'execution_time': 110},
            {'deployment_id': 'deploy-3', 'status': 'failed', 'execution_time': 200}
        ]
        
        metrics = deployment.get_deployment_metrics(days=7)
        
        assert 'total_deployments' in metrics
        assert 'success_rate' in metrics
        assert 'average_deployment_time' in metrics
        assert 'rollback_count' in metrics
        
        assert metrics['total_deployments'] == 3
        assert 0 <= metrics['success_rate'] <= 1
        assert metrics['average_deployment_time'] > 0


class TestQuantumABTest:
    """Test cases for QuantumABTest class."""
    
    def test_initialization(self):
        """Test QuantumABTest initialization."""
        ab_test = QuantumABTest(
            test_name="VQE_optimization_test",
            description="Comparing VQE optimizers"
        )
        
        assert ab_test.test_name == "VQE_optimization_test"
        assert "optimization" in ab_test.description
        assert isinstance(ab_test.algorithms, dict)
        assert isinstance(ab_test.results, dict)
    
    def test_add_algorithm(self):
        """Test adding algorithms to A/B test."""
        ab_test = QuantumABTest("optimizer_test")
        
        algorithm_a = {
            'name': 'COBYLA_VQE',
            'circuit': 'vqe_circuit',
            'optimizer': 'COBYLA',
            'parameters': {'maxiter': 1000}
        }
        
        algorithm_b = {
            'name': 'SPSA_VQE',
            'circuit': 'vqe_circuit',
            'optimizer': 'SPSA',
            'parameters': {'maxiter': 1000}
        }
        
        ab_test.add_algorithm('A', algorithm_a)
        ab_test.add_algorithm('B', algorithm_b)
        
        assert 'A' in ab_test.algorithms
        assert 'B' in ab_test.algorithms
        assert ab_test.algorithms['A']['optimizer'] == 'COBYLA'
        assert ab_test.algorithms['B']['optimizer'] == 'SPSA'
    
    def test_run_test_mock(self):
        """Test running A/B test with mocked execution."""
        ab_test = QuantumABTest("mock_test")
        
        # Add test algorithms
        ab_test.add_algorithm('A', {'name': 'Algo_A', 'circuit': 'test_circuit_a'})
        ab_test.add_algorithm('B', {'name': 'Algo_B', 'circuit': 'test_circuit_b'})
        
        # Mock the execution function
        def mock_execute(algorithm, shots, backend):
            # Return different results for different algorithms
            if 'A' in algorithm['name']:
                return {'fidelity': 0.92, 'execution_time': 120, 'cost': 5.0}
            else:
                return {'fidelity': 0.88, 'execution_time': 100, 'cost': 4.5}
        
        with patch.object(ab_test, '_execute_algorithm', side_effect=mock_execute):
            ab_test.run_test(shots=1000, iterations=5)
        
        assert 'A' in ab_test.results
        assert 'B' in ab_test.results
        assert isinstance(ab_test.results['A'], ABTestResult)
        assert isinstance(ab_test.results['B'], ABTestResult)
        assert len(ab_test.results['A'].measurements) == 5
        assert len(ab_test.results['B'].measurements) == 5
    
    def test_determine_winner(self):
        """Test winner determination in A/B test."""
        ab_test = QuantumABTest("winner_test")
        
        # Create mock results with clear winner
        results_a = ABTestResult(
            algorithm_name='Algorithm_A',
            measurements=[0.95, 0.94, 0.96, 0.95, 0.94],  # High fidelity
            sample_size=5,
            execution_time_seconds=120,
            cost_usd=5.0,
            success_rate=1.0
        )
        
        results_b = ABTestResult(
            algorithm_name='Algorithm_B', 
            measurements=[0.85, 0.84, 0.86, 0.85, 0.84],  # Lower fidelity
            sample_size=5,
            execution_time_seconds=100,
            cost_usd=4.0,
            success_rate=1.0
        )
        
        ab_test.results = {'A': results_a, 'B': results_b}
        
        analysis = ab_test.determine_winner(
            confidence_level=0.95,
            minimum_difference=0.05
        )
        
        assert isinstance(analysis, ABTestAnalysis)
        assert analysis.winner in ['A', 'B', 'inconclusive']
        assert 0 <= analysis.confidence_level <= 1
        assert analysis.p_value >= 0
        assert isinstance(analysis.recommendation, str)
    
    def test_get_test_summary(self):
        """Test A/B test summary generation."""
        ab_test = QuantumABTest("summary_test", description="Test summary generation")
        
        # Add mock results
        ab_test.results = {
            'A': ABTestResult(
                algorithm_name='Fast_Algorithm',
                measurements=[0.90, 0.91, 0.89],
                sample_size=3,
                execution_time_seconds=60,
                cost_usd=2.0,
                success_rate=1.0
            ),
            'B': ABTestResult(
                algorithm_name='Accurate_Algorithm',
                measurements=[0.95, 0.96, 0.94],
                sample_size=3,
                execution_time_seconds=120,
                cost_usd=5.0,
                success_rate=1.0
            )
        }
        
        summary = ab_test.get_test_summary()
        
        assert 'test_name' in summary
        assert 'description' in summary
        assert 'algorithms_tested' in summary
        assert 'results_summary' in summary
        assert 'winner_analysis' in summary
        
        assert summary['test_name'] == "summary_test"
        assert len(summary['algorithms_tested']) == 2
        assert 'A' in summary['results_summary']
        assert 'B' in summary['results_summary']
    
    def test_statistical_analysis(self):
        """Test statistical analysis methods."""
        ab_test = QuantumABTest("stats_test")
        
        # Create results with known statistical properties
        measurements_a = [0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91]
        measurements_b = [0.85, 0.86, 0.84, 0.87, 0.83, 0.85, 0.86]
        
        # Test t-test calculation
        t_stat, p_value = ab_test._calculate_t_test(measurements_a, measurements_b)
        
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert p_value >= 0
        assert p_value <= 1
        
        # Test effect size calculation
        effect_size = ab_test._calculate_effect_size(measurements_a, measurements_b)
        
        assert isinstance(effect_size, float)
        # Effect size should be positive since measurements_a > measurements_b
        assert effect_size > 0
    
    def test_cost_benefit_analysis(self):
        """Test cost-benefit analysis in A/B testing."""
        ab_test = QuantumABTest("cost_benefit_test")
        
        # Create results with cost-accuracy tradeoff
        results_fast_cheap = ABTestResult(
            algorithm_name='Fast_Cheap',
            measurements=[0.85, 0.84, 0.86],
            sample_size=3,
            execution_time_seconds=30,
            cost_usd=1.0,
            success_rate=1.0
        )
        
        results_slow_accurate = ABTestResult(
            algorithm_name='Slow_Accurate',
            measurements=[0.95, 0.96, 0.94],
            sample_size=3,
            execution_time_seconds=180,
            cost_usd=8.0,
            success_rate=1.0
        )
        
        ab_test.results = {'fast': results_fast_cheap, 'accurate': results_slow_accurate}
        
        cost_benefit = ab_test.analyze_cost_benefit()
        
        assert 'algorithms' in cost_benefit
        assert 'cost_per_fidelity' in cost_benefit
        assert 'time_per_fidelity' in cost_benefit
        assert 'efficiency_ranking' in cost_benefit
        
        # Should calculate cost and time efficiency
        assert len(cost_benefit['cost_per_fidelity']) == 2
        assert len(cost_benefit['efficiency_ranking']) == 2


@pytest.mark.performance
class TestDeploymentPerformance:
    """Performance tests for deployment functionality."""
    
    def test_large_ab_test_performance(self):
        """Test A/B test performance with large sample sizes."""
        import time
        
        ab_test = QuantumABTest("performance_test")
        
        # Create large datasets
        large_measurements_a = [0.90 + (i % 10) / 100 for i in range(100)]
        large_measurements_b = [0.85 + (i % 10) / 100 for i in range(100)]
        
        ab_test.results = {
            'A': ABTestResult('Algo_A', large_measurements_a, 100, 120, 5.0, 1.0),
            'B': ABTestResult('Algo_B', large_measurements_b, 100, 110, 4.5, 1.0)
        }
        
        start_time = time.time()
        analysis = ab_test.determine_winner()
        analysis_time = time.time() - start_time
        
        # Should analyze large datasets quickly
        assert analysis_time < 1.0
        assert isinstance(analysis, ABTestAnalysis)
    
    def test_deployment_history_performance(self):
        """Test deployment history performance with many deployments."""
        import time
        
        config = DeploymentConfig()
        deployment = QuantumDeployment("perf-test-service", config)
        
        # Create large deployment history
        start_time = time.time()
        for i in range(200):
            deployment.deployment_history.append({
                'deployment_id': f'deploy-{i}',
                'version': f'1.0.{i}',
                'timestamp': datetime.now() - timedelta(hours=i),
                'status': 'success' if i % 10 != 0 else 'failed',
                'execution_time': 120 + (i % 20)
            })
        
        history_creation_time = time.time() - start_time
        
        # Test metrics calculation performance
        start_time = time.time()
        metrics = deployment.get_deployment_metrics(days=30)
        metrics_time = time.time() - start_time
        
        # Should handle large history efficiently
        assert history_creation_time < 1.0
        assert metrics_time < 0.5
        assert metrics['total_deployments'] == 200


@pytest.mark.integration
class TestDeploymentIntegration:
    """Integration tests for deployment functionality."""
    
    def test_deployment_with_monitoring(self):
        """Test deployment integration with monitoring."""
        config = DeploymentConfig()
        deployment = QuantumDeployment("integration-service", config)
        
        # Mock algorithm deployment
        algorithm = {
            'name': 'Integration_Test_VQE',
            'version': '1.0.0',
            'circuit': 'mock_integration_circuit'
        }
        
        deploy_result = deployment.deploy_blue_green(algorithm)
        health_status = deployment.check_health()
        
        assert deploy_result['status'] in ['success', 'failed', 'pending']
        assert health_status['status'] in ['healthy', 'unhealthy', 'degraded']
    
    def test_ab_test_with_real_execution_mock(self):
        """Test A/B test with mocked real quantum execution."""
        ab_test = QuantumABTest("integration_ab_test")
        
        # Add algorithms
        ab_test.add_algorithm('optimizer_a', {
            'name': 'COBYLA_VQE',
            'optimizer': 'COBYLA',
            'circuit': 'h2_vqe_circuit'
        })
        
        ab_test.add_algorithm('optimizer_b', {
            'name': 'SPSA_VQE', 
            'optimizer': 'SPSA',
            'circuit': 'h2_vqe_circuit'
        })
        
        # Mock quantum execution with realistic results
        def mock_quantum_execute(algorithm, shots, backend):
            if 'COBYLA' in algorithm['name']:
                return {
                    'fidelity': 0.93 + (hash(str(shots)) % 100) / 1000,  # Add some variance
                    'execution_time': 150,
                    'cost': 6.0
                }
            else:
                return {
                    'fidelity': 0.89 + (hash(str(shots)) % 100) / 1000,
                    'execution_time': 120,
                    'cost': 4.8
                }
        
        with patch.object(ab_test, '_execute_algorithm', side_effect=mock_quantum_execute):
            ab_test.run_test(shots=1000, iterations=3)
        
        analysis = ab_test.determine_winner()
        summary = ab_test.get_test_summary()
        
        assert len(ab_test.results) == 2
        assert isinstance(analysis, ABTestAnalysis)
        assert 'winner_analysis' in summary
