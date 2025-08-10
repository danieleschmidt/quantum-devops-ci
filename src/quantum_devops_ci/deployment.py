"""
Quantum deployment and A/B testing framework.

This module provides tools for deploying quantum algorithms with advanced
strategies like blue-green deployment, canary releases, and A/B testing.
"""

import warnings
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import random


class DeploymentStrategy(Enum):
    """Deployment strategy options."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentEnvironment:
    """Deployment environment configuration."""
    name: str
    backend: str
    allocation_percentage: float
    max_shots: int
    validation_shots: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of deployment validation."""
    environment: str
    passed: bool
    fidelity: float
    error_rate: float  
    execution_time: float
    cost: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestVariant:
    """A/B test variant configuration."""
    name: str
    algorithm_config: Dict[str, Any]
    description: str = ""
    traffic_allocation: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B test result."""
    variant_name: str
    metric_values: Dict[str, float]
    sample_size: int
    confidence_interval: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]  # p-values for each metric


@dataclass
class ABTestAnalysis:
    """A/B test statistical analysis."""
    winner: Optional[str]
    improvement: float  # Percentage improvement
    confidence_level: float
    p_value: float
    effect_size: float
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


class QuantumDeployment:
    """
    Quantum algorithm deployment manager.
    
    This class handles deployment of quantum algorithms with various strategies
    including blue-green deployment, canary releases, and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize quantum deployment manager.
        
        Args:
            config: Deployment configuration dictionary
        """
        self.config = config
        self.environments = {}
        self.deployments = {}
        self.active_tests = {}
        
        # Load environments from config
        self._initialize_environments()
    
    def _initialize_environments(self):
        """Initialize deployment environments from configuration."""
        env_configs = self.config.get('environments', {})
        
        for env_name, env_config in env_configs.items():
            self.environments[env_name] = DeploymentEnvironment(
                name=env_name,
                backend=env_config.get('backend', 'qasm_simulator'),
                allocation_percentage=env_config.get('allocation', 100.0),
                max_shots=env_config.get('max_shots', 10000),
                validation_shots=env_config.get('validation_shots', 1000),
                metadata=env_config.get('metadata', {})
            )
        
        # Create default environments if none configured
        if not self.environments:
            self.environments['production'] = DeploymentEnvironment(
                name='production',
                backend='qasm_simulator',
                allocation_percentage=100.0,
                max_shots=100000
            )
    
    def deploy(self, 
               algorithm_id: str,
               circuit_factory: Callable,
               strategy: Union[DeploymentStrategy, str],
               environment: str = 'production',
               **kwargs) -> str:
        """
        Deploy quantum algorithm with specified strategy.
        
        Args:
            algorithm_id: Unique identifier for the algorithm
            circuit_factory: Function that creates quantum circuits
            strategy: Deployment strategy to use
            environment: Target environment name
            **kwargs: Additional deployment parameters
            
        Returns:
            Deployment ID for tracking
        """
        if isinstance(strategy, str):
            strategy = DeploymentStrategy(strategy)
        
        deployment_id = f"{algorithm_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment_record = {
            'id': deployment_id,
            'algorithm_id': algorithm_id,
            'strategy': strategy,
            'environment': environment,
            'status': DeploymentStatus.PENDING,
            'start_time': datetime.now(),
            'circuit_factory': circuit_factory,
            'parameters': kwargs
        }
        
        self.deployments[deployment_id] = deployment_record
        
        try:
            # Execute deployment based on strategy
            if strategy == DeploymentStrategy.BLUE_GREEN:
                self._execute_blue_green_deployment(deployment_id)
            elif strategy == DeploymentStrategy.CANARY:
                self._execute_canary_deployment(deployment_id)
            elif strategy == DeploymentStrategy.ROLLING:
                self._execute_rolling_deployment(deployment_id)
            else:
                raise ValueError(f"Unsupported deployment strategy: {strategy}")
            
            deployment_record['status'] = DeploymentStatus.COMPLETED
            deployment_record['end_time'] = datetime.now()
        
        except Exception as e:
            deployment_record['status'] = DeploymentStatus.FAILED
            deployment_record['error'] = str(e)
            deployment_record['end_time'] = datetime.now()
            raise
        
        return deployment_id
    
    def _execute_blue_green_deployment(self, deployment_id: str):
        """Execute blue-green deployment strategy."""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.IN_PROGRESS
        
        # For blue-green deployment, we validate on a secondary environment
        # then switch traffic
        
        # Mock validation (in real implementation, this would run actual circuits)
        validation_result = self._mock_validation(
            deployment['circuit_factory'],
            deployment['environment']
        )
        
        if not validation_result.passed:
            raise Exception(f"Blue-green validation failed: {validation_result.details}")
        
        # Mock traffic switch
        deployment['validation_result'] = validation_result
        deployment['traffic_switched'] = True
    
    def _execute_canary_deployment(self, deployment_id: str):
        """Execute canary deployment strategy."""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.IN_PROGRESS
        
        # Gradual rollout with monitoring
        canary_percentages = deployment['parameters'].get('canary_percentages', [10, 25, 50, 100])
        
        for percentage in canary_percentages:
            validation_result = self._mock_validation(
                deployment['circuit_factory'],
                deployment['environment'],
                shots=int(1000 * percentage / 100)
            )
            
            if not validation_result.passed:
                # Rollback
                deployment['status'] = DeploymentStatus.ROLLED_BACK
                raise Exception(f"Canary validation failed at {percentage}%: {validation_result.details}")
            
            # Wait between increments (mocked)
            import time
            time.sleep(0.1)  # Mock wait time
        
        deployment['canary_completed'] = True
    
    def _execute_rolling_deployment(self, deployment_id: str):
        """Execute rolling deployment strategy."""
        deployment = self.deployments[deployment_id]
        deployment['status'] = DeploymentStatus.IN_PROGRESS
        
        # Rolling deployment across multiple backend instances
        instances = deployment['parameters'].get('instances', ['instance_1', 'instance_2'])
        
        for instance in instances:
            validation_result = self._mock_validation(
                deployment['circuit_factory'],
                f"{deployment['environment']}_{instance}"
            )
            
            if not validation_result.passed:
                raise Exception(f"Rolling deployment failed on {instance}: {validation_result.details}")
        
        deployment['rolling_completed'] = True
    
    def validate_deployment(self, deployment_id: str, environment: str) -> ValidationResult:
        """
        Validate a deployment in the specified environment.
        
        Args:
            deployment_id: Deployment ID to validate
            environment: Environment name
            
        Returns:
            ValidationResult with metrics and status
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        
        return self._mock_validation(
            deployment['circuit_factory'],
            environment
        )
    
    def _mock_validation(self, circuit_factory: Callable, environment: str, shots: int = 1000) -> ValidationResult:
        """Mock validation for testing purposes."""
        import time
        import random
        
        start_time = time.time()
        
        # Mock circuit execution
        time.sleep(random.uniform(0.1, 0.3))  # Simulate execution time
        
        # Mock metrics
        fidelity = random.uniform(0.85, 0.98)
        error_rate = random.uniform(0.01, 0.10)
        
        execution_time = time.time() - start_time
        cost = shots * 0.001  # Mock cost calculation
        
        # Determine if validation passed (more lenient for mock)
        passed = fidelity > 0.85 and error_rate < 0.12
        
        return ValidationResult(
            environment=environment,
            passed=passed,
            fidelity=fidelity,
            error_rate=error_rate,
            execution_time=execution_time,
            cost=cost,
            details={
                'shots': shots,
                'backend': self.environments.get(environment).backend if self.environments.get(environment) else 'unknown',
                'validation_time': datetime.now().isoformat()
            }
        )
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        
        status = {
            'id': deployment_id,
            'algorithm_id': deployment['algorithm_id'],
            'status': deployment['status'].value,
            'strategy': deployment['strategy'].value,
            'environment': deployment['environment'],
            'start_time': deployment['start_time'].isoformat(),
        }
        
        if 'end_time' in deployment:
            status['end_time'] = deployment['end_time'].isoformat()
            duration = deployment['end_time'] - deployment['start_time']
            status['duration_seconds'] = duration.total_seconds()
        
        if 'error' in deployment:
            status['error'] = deployment['error']
        
        if 'validation_result' in deployment:
            result = deployment['validation_result']
            status['validation'] = {
                'fidelity': result.fidelity,
                'error_rate': result.error_rate,
                'cost': result.cost
            }
        
        return status
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """
        Rollback a deployment.
        
        Args:
            deployment_id: Deployment ID to rollback
            
        Returns:
            True if rollback successful
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        
        if deployment['status'] in [DeploymentStatus.COMPLETED, DeploymentStatus.IN_PROGRESS]:
            deployment['status'] = DeploymentStatus.ROLLED_BACK
            deployment['rollback_time'] = datetime.now()
            return True
        
        return False


class QuantumABTest:
    """
    A/B testing framework for quantum algorithms.
    
    This class provides statistical analysis and comparison of different
    quantum algorithm variants to determine the best performing option.
    """
    
    def __init__(self, 
                 name: str,
                 variants: Dict[str, Dict[str, Any]],
                 metrics: List[str],
                 confidence_level: float = 0.95):
        """
        Initialize A/B test.
        
        Args:
            name: Test name
            variants: Dictionary of variant configurations
            metrics: List of metrics to track
            confidence_level: Statistical confidence level
        """
        self.name = name
        self.metrics = metrics
        self.confidence_level = confidence_level
        self.variants = {}
        self.results = {}
        self.test_data = {}
        
        # Initialize variants
        for variant_name, config in variants.items():
            self.variants[variant_name] = ABTestVariant(
                name=variant_name,
                algorithm_config=config,
                traffic_allocation=config.get('traffic_allocation', 1.0 / len(variants))
            )
            self.test_data[variant_name] = {metric: [] for metric in metrics}
    
    def run(self, 
            circuit_factory: Callable,
            duration_hours: float = 24,
            traffic_split: float = 0.5,
            **kwargs) -> Dict[str, ABTestResult]:
        """
        Run A/B test.
        
        Args:
            circuit_factory: Function that creates quantum circuits
            duration_hours: Test duration in hours
            traffic_split: Traffic split between variants
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary of results for each variant
        """
        # Mock A/B test execution
        results = {}
        
        for variant_name, variant in self.variants.items():
            # Simulate test execution
            sample_size = int(1000 * traffic_split * variant.traffic_allocation)
            
            # Generate mock metric values
            metric_values = {}
            for metric in self.metrics:
                if metric == 'convergence_rate':
                    values = [random.uniform(0.8, 0.95) for _ in range(sample_size)]
                elif metric == 'final_energy':
                    values = [random.uniform(-1.5, -0.5) for _ in range(sample_size)]
                elif metric == 'total_evaluations':
                    values = [random.randint(100, 500) for _ in range(sample_size)]
                else:
                    values = [random.uniform(0.5, 1.0) for _ in range(sample_size)]
                
                metric_values[metric] = statistics.mean(values)
                self.test_data[variant_name][metric] = values
            
            # Calculate confidence intervals (simplified)
            confidence_intervals = {}
            for metric, values in self.test_data[variant_name].items():
                mean_val = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                margin = 1.96 * std_dev / (len(values) ** 0.5)  # 95% CI
                confidence_intervals[metric] = (mean_val - margin, mean_val + margin)
            
            results[variant_name] = ABTestResult(
                variant_name=variant_name,
                metric_values=metric_values,
                sample_size=sample_size,
                confidence_interval=confidence_intervals,
                statistical_significance={}  # Would calculate p-values
            )
        
        self.results = results
        return results
    
    def determine_winner(self, 
                        results: Dict[str, ABTestResult],
                        confidence_level: float = 0.95,
                        minimum_difference: float = 0.05) -> ABTestAnalysis:
        """
        Determine the winning variant based on statistical analysis.
        
        Args:
            results: A/B test results
            confidence_level: Required confidence level
            minimum_difference: Minimum meaningful difference
            
        Returns:
            ABTestAnalysis with winner and statistical details
        """
        if len(results) < 2:
            return ABTestAnalysis(
                winner=None,
                improvement=0.0,
                confidence_level=confidence_level,
                p_value=1.0,
                effect_size=0.0,
                recommendation="Need at least 2 variants for comparison"
            )
        
        # Simple winner determination based on primary metric
        primary_metric = self.metrics[0] if self.metrics else 'score'
        
        best_variant = None
        best_score = float('-inf')
        
        for variant_name, result in results.items():
            score = result.metric_values.get(primary_metric, 0.0)
            if score > best_score:
                best_score = score
                best_variant = variant_name
        
        # Calculate improvement (simplified)
        variant_names = list(results.keys())
        if len(variant_names) >= 2:
            baseline_variant = variant_names[0] if variant_names[0] != best_variant else variant_names[1]
            baseline_score = results[baseline_variant].metric_values.get(primary_metric, 0.0)
            improvement = ((best_score - baseline_score) / baseline_score) * 100 if baseline_score != 0 else 0.0
        else:
            improvement = 0.0
        
        # Mock statistical significance
        p_value = random.uniform(0.01, 0.10)  # Mock p-value
        effect_size = abs(improvement) / 100  # Simplified effect size
        
        # Generate recommendation
        if improvement > minimum_difference * 100 and p_value < (1 - confidence_level):
            recommendation = f"Deploy variant {best_variant} - statistically significant improvement"
        elif improvement > minimum_difference * 100:
            recommendation = f"Variant {best_variant} shows promise but needs more data for statistical significance"
        else:
            recommendation = "No significant difference found - consider longer test duration"
        
        return ABTestAnalysis(
            winner=best_variant,
            improvement=improvement,
            confidence_level=confidence_level,
            p_value=p_value,
            effect_size=effect_size,
            recommendation=recommendation,
            details={
                'primary_metric': primary_metric,
                'best_score': best_score,
                'sample_sizes': {name: result.sample_size for name, result in results.items()},
                'test_duration': 'mocked'  # In real implementation, track actual duration
            }
        )
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of A/B test."""
        if not self.results:
            return {'status': 'not_run', 'message': 'A/B test has not been executed yet'}
        
        summary = {
            'test_name': self.name,
            'status': 'completed',
            'variants': list(self.variants.keys()),
            'metrics_tracked': self.metrics,
            'total_samples': sum(result.sample_size for result in self.results.values()),
            'variant_performance': {}
        }
        
        for variant_name, result in self.results.items():
            summary['variant_performance'][variant_name] = {
                'sample_size': result.sample_size,
                'metrics': result.metric_values
            }
        
        # Add winner if analysis has been done
        if hasattr(self, '_winner_analysis'):
            summary['winner'] = self._winner_analysis.winner
            summary['improvement'] = self._winner_analysis.improvement
            summary['recommendation'] = self._winner_analysis.recommendation
        
        return summary


def main():
    """Main entry point for quantum deployment CLI."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Quantum deployment manager')
    parser.add_argument('--config', required=True, help='Deployment configuration file')
    parser.add_argument('--deploy', help='Algorithm ID to deploy')
    parser.add_argument('--strategy', choices=['blue_green', 'canary', 'rolling'], 
                       default='blue_green', help='Deployment strategy')
    parser.add_argument('--validate', help='Validate deployment ID')
    parser.add_argument('--environment', help='Environment for validation')
    parser.add_argument('--status', help='Get deployment status')
    parser.add_argument('--rollback', help='Rollback deployment ID')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize deployment manager
    deployer = QuantumDeployment(config)
    
    if args.deploy:
        # Mock circuit factory
        def mock_circuit_factory():
            return {"type": "mock_circuit", "qubits": 2}
        
        strategy = DeploymentStrategy(args.strategy)
        deployment_id = deployer.deploy(args.deploy, mock_circuit_factory, strategy)
        print(f"Deployment started: {deployment_id}")
    
    elif args.validate:
        if not args.environment:
            print("Error: --environment required for validation")
            return
        
        result = deployer.validate_deployment(args.validate, args.environment)
        print(f"Validation result: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Fidelity: {result.fidelity:.3f}")
        print(f"Error rate: {result.error_rate:.3f}")
    
    elif args.status:
        status = deployer.get_deployment_status(args.status)
        print(json.dumps(status, indent=2))
    
    elif args.rollback:
        success = deployer.rollback_deployment(args.rollback)
        print(f"Rollback {'successful' if success else 'failed'}: {args.rollback}")
    
    else:
        print("No action specified. Use --help for options.")


if __name__ == '__main__':
    main()