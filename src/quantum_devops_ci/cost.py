"""
Cost optimization for quantum computing workloads.

This module provides tools for optimizing quantum computing costs through
intelligent resource allocation, provider selection, and usage forecasting.
"""

import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq


class ProviderType(Enum):
    """Quantum cloud provider types."""
    IBMQ = "ibmq"
    AWS_BRAKET = "aws_braket"
    GOOGLE_QUANTUM = "google_quantum"
    RIGETTI = "rigetti"
    IONQ = "ionq"


@dataclass
class CostModel:
    """Cost model for quantum provider/backend."""
    provider: ProviderType
    backend_name: str
    cost_per_shot: float  # USD
    cost_per_minute: Optional[float] = None  # USD, for time-based billing
    minimum_cost: float = 0.0  # Minimum charge per job
    setup_cost: float = 0.0  # One-time setup cost
    bulk_discount_threshold: int = 10000  # Shots threshold for bulk discount
    bulk_discount_rate: float = 0.0  # Discount rate for bulk usage
    priority_multiplier: float = 1.0  # Cost multiplier for priority access


@dataclass
class UsageQuota:
    """Usage quota and limits."""
    provider: ProviderType
    monthly_shot_limit: int
    monthly_cost_limit: float
    daily_shot_limit: Optional[int] = None
    daily_cost_limit: Optional[float] = None
    current_monthly_shots: int = 0
    current_monthly_cost: float = 0.0
    current_daily_shots: int = 0
    current_daily_cost: float = 0.0
    
    def has_capacity(self, shots: int, cost: float) -> bool:
        """Check if quota has capacity for additional usage."""
        monthly_shots_ok = (self.current_monthly_shots + shots) <= self.monthly_shot_limit
        monthly_cost_ok = (self.current_monthly_cost + cost) <= self.monthly_cost_limit
        
        daily_shots_ok = True
        daily_cost_ok = True
        
        if self.daily_shot_limit:
            daily_shots_ok = (self.current_daily_shots + shots) <= self.daily_shot_limit
        if self.daily_cost_limit:
            daily_cost_ok = (self.current_daily_cost + cost) <= self.daily_cost_limit
        
        return monthly_shots_ok and monthly_cost_ok and daily_shots_ok and daily_cost_ok
    
    def remaining_monthly_shots(self) -> int:
        """Get remaining monthly shot quota."""
        return max(0, self.monthly_shot_limit - self.current_monthly_shots)
    
    def remaining_monthly_budget(self) -> float:
        """Get remaining monthly budget."""
        return max(0.0, self.monthly_cost_limit - self.current_monthly_cost)


@dataclass
class ExperimentSpec:
    """Specification for quantum experiment."""
    id: str
    circuit: Any  # Quantum circuit
    shots: int
    priority: str = "medium"  # low, medium, high
    backend_preferences: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    max_cost: Optional[float] = None
    description: str = ""


@dataclass 
class CostOptimizationResult:
    """Result of cost optimization."""
    original_cost: float
    optimized_cost: float
    savings: float
    savings_percentage: float
    optimized_assignments: List[Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.savings = self.original_cost - self.optimized_cost
        self.savings_percentage = (self.savings / self.original_cost * 100) if self.original_cost > 0 else 0.0


class CostOptimizer:
    """Quantum computing cost optimization engine."""
    
    def __init__(self, 
                 monthly_budget: float = 1000.0,
                 priority_weights: Optional[Dict[str, float]] = None):
        """
        Initialize cost optimizer.
        
        Args:
            monthly_budget: Monthly budget in USD
            priority_weights: Weights for different priority levels
        """
        self.monthly_budget = monthly_budget
        self.priority_weights = priority_weights or {
            'low': 0.1,
            'medium': 0.6, 
            'high': 0.3
        }
        
        # Initialize cost models and quotas
        self.cost_models = {}
        self.quotas = {}
        self.usage_history = []
        
        self._initialize_cost_models()
        self._initialize_quotas()
    
    def _initialize_cost_models(self):
        """Initialize cost models for different providers."""
        # IBM Quantum cost models
        self.cost_models['ibmq_qasm_simulator'] = CostModel(
            provider=ProviderType.IBMQ,
            backend_name='ibmq_qasm_simulator',
            cost_per_shot=0.0,  # Free simulator
            minimum_cost=0.0
        )
        
        self.cost_models['ibmq_manhattan'] = CostModel(
            provider=ProviderType.IBMQ,
            backend_name='ibmq_manhattan',
            cost_per_shot=0.001,
            minimum_cost=0.10,
            bulk_discount_threshold=10000,
            bulk_discount_rate=0.1
        )
        
        # AWS Braket cost models
        self.cost_models['aws_sv1'] = CostModel(
            provider=ProviderType.AWS_BRAKET,
            backend_name='aws_sv1',
            cost_per_shot=0.00075,
            minimum_cost=0.15,
            bulk_discount_threshold=50000,
            bulk_discount_rate=0.15
        )
        
        self.cost_models['aws_tn1'] = CostModel(
            provider=ProviderType.AWS_BRAKET,
            backend_name='aws_tn1',
            cost_per_shot=0.000275,
            cost_per_minute=0.25,
            minimum_cost=0.30
        )
        
        # IonQ cost model
        self.cost_models['ionq_qpu'] = CostModel(
            provider=ProviderType.IONQ,
            backend_name='ionq_qpu',
            cost_per_shot=0.01,
            minimum_cost=1.00,
            setup_cost=0.50
        )
    
    def _initialize_quotas(self):
        """Initialize usage quotas for providers."""
        # Distribute budget across providers based on typical usage
        ibmq_budget = self.monthly_budget * 0.4
        aws_budget = self.monthly_budget * 0.35 
        ionq_budget = self.monthly_budget * 0.25
        
        self.quotas[ProviderType.IBMQ] = UsageQuota(
            provider=ProviderType.IBMQ,
            monthly_shot_limit=int(ibmq_budget / 0.001),  # Based on avg cost
            monthly_cost_limit=ibmq_budget,
            daily_cost_limit=ibmq_budget / 30
        )
        
        self.quotas[ProviderType.AWS_BRAKET] = UsageQuota(
            provider=ProviderType.AWS_BRAKET,
            monthly_shot_limit=int(aws_budget / 0.0005),
            monthly_cost_limit=aws_budget,
            daily_cost_limit=aws_budget / 30
        )
        
        self.quotas[ProviderType.IONQ] = UsageQuota(
            provider=ProviderType.IONQ,
            monthly_shot_limit=int(ionq_budget / 0.01),
            monthly_cost_limit=ionq_budget,
            daily_cost_limit=ionq_budget / 30
        )
    
    def calculate_job_cost(self, 
                          shots: int, 
                          backend: str,
                          execution_time_minutes: Optional[float] = None) -> float:
        """Calculate cost for a specific job."""
        if backend not in self.cost_models:
            raise ValueError(f"Unknown backend: {backend}")
        
        model = self.cost_models[backend]
        
        # Base cost from shots
        shot_cost = shots * model.cost_per_shot
        
        # Time-based cost if applicable
        time_cost = 0.0
        if model.cost_per_minute and execution_time_minutes:
            time_cost = execution_time_minutes * model.cost_per_minute
        
        # Total base cost
        base_cost = shot_cost + time_cost + model.setup_cost
        
        # Apply bulk discount if applicable
        if shots >= model.bulk_discount_threshold:
            discount = base_cost * model.bulk_discount_rate
            base_cost -= discount
        
        # Apply minimum cost
        final_cost = max(base_cost, model.minimum_cost)
        
        return final_cost
    
    def optimize_experiments(self, 
                            experiments: List[Dict[str, Any]],
                            constraints: Optional[Dict[str, Any]] = None) -> CostOptimizationResult:
        """
        Optimize experiment assignments to minimize costs.
        
        Args:
            experiments: List of experiment specifications
            constraints: Additional constraints (deadline, max_cost, etc.)
            
        Returns:
            CostOptimizationResult with optimized assignments
        """
        if constraints is None:
            constraints = {}
        
        # Convert to ExperimentSpec objects
        experiment_specs = []
        for exp in experiments:
            spec = ExperimentSpec(
                id=exp.get('id', f'exp_{len(experiment_specs)}'),
                circuit=exp.get('circuit', None),
                shots=exp.get('shots', 1000),
                priority=exp.get('priority', 'medium'),
                backend_preferences=exp.get('backend_preferences', []),
                max_cost=exp.get('max_cost')
            )
            experiment_specs.append(spec)
        
        # Calculate original cost (naive assignment)
        original_assignments = []
        original_cost = 0.0
        
        for spec in experiment_specs:
            # Use first available backend or default
            backend = spec.backend_preferences[0] if spec.backend_preferences else 'ibmq_qasm_simulator'
            if backend not in self.cost_models:
                backend = 'ibmq_qasm_simulator'
            
            cost = self.calculate_job_cost(spec.shots, backend)
            original_assignments.append({
                'experiment_id': spec.id,
                'backend': backend,
                'cost': cost,
                'shots': spec.shots
            })
            original_cost += cost
        
        # Optimize assignments
        optimized_assignments = self._optimize_assignments(experiment_specs, constraints)
        optimized_cost = sum(assignment['cost'] for assignment in optimized_assignments)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(experiment_specs, optimized_assignments)
        
        return CostOptimizationResult(
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            savings=0.0,  # Will be calculated in __post_init__
            savings_percentage=0.0,
            optimized_assignments=optimized_assignments,
            recommendations=recommendations
        )
    
    def _optimize_assignments(self, 
                             experiments: List[ExperimentSpec],
                             constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize backend assignments for experiments."""
        assignments = []
        
        # Sort experiments by priority and cost sensitivity
        def priority_score(exp):
            base_score = {'high': 3, 'medium': 2, 'low': 1}.get(exp.priority, 1)
            # Higher shot count experiments get slightly higher priority for bulk discounts
            shot_bonus = min(exp.shots / 10000, 1.0)
            return base_score + shot_bonus
        
        sorted_experiments = sorted(experiments, key=priority_score, reverse=True)
        
        for exp in sorted_experiments:
            best_assignment = self._find_best_backend_assignment(exp, constraints)
            assignments.append(best_assignment)
        
        return assignments
    
    def _find_best_backend_assignment(self, 
                                     experiment: ExperimentSpec,
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Find best backend assignment for a single experiment."""
        candidate_backends = experiment.backend_preferences or list(self.cost_models.keys())
        
        best_backend = None
        best_cost = float('inf')
        
        for backend in candidate_backends:
            if backend not in self.cost_models:
                continue
            
            cost = self.calculate_job_cost(experiment.shots, backend)
            
            # Check quota constraints
            model = self.cost_models[backend]
            quota = self.quotas.get(model.provider)
            
            if quota and not quota.has_capacity(experiment.shots, cost):
                continue
            
            # Check experiment cost limit
            if experiment.max_cost and cost > experiment.max_cost:
                continue
            
            # Check global constraints
            if 'max_cost_per_experiment' in constraints and cost > constraints['max_cost_per_experiment']:
                continue
            
            if cost < best_cost:
                best_cost = cost
                best_backend = backend
        
        # Fallback to free simulator if no backend fits constraints
        if best_backend is None:
            best_backend = 'ibmq_qasm_simulator'
            best_cost = self.calculate_job_cost(experiment.shots, best_backend)
        
        return {
            'experiment_id': experiment.id,
            'backend': best_backend,
            'cost': best_cost,
            'shots': experiment.shots,
            'priority': experiment.priority
        }
    
    def _generate_recommendations(self, 
                                 experiments: List[ExperimentSpec],
                                 assignments: List[Dict[str, Any]]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        # Check for bulk discount opportunities
        backend_usage = {}
        for assignment in assignments:
            backend = assignment['backend']
            shots = assignment['shots']
            
            if backend not in backend_usage:
                backend_usage[backend] = 0
            backend_usage[backend] += shots
        
        for backend, total_shots in backend_usage.items():
            if backend in self.cost_models:
                model = self.cost_models[backend]
                if total_shots < model.bulk_discount_threshold:
                    needed = model.bulk_discount_threshold - total_shots
                    recommendations.append(
                        f"Add {needed} more shots to {backend} to qualify for {model.bulk_discount_rate:.1%} bulk discount"
                    )
        
        # Check for simulator usage opportunities
        hardware_experiments = [a for a in assignments if 'simulator' not in a['backend']]
        if len(hardware_experiments) > len(assignments) * 0.8:
            recommendations.append(
                "Consider using simulators for initial testing to reduce costs"
            )
        
        # Budget utilization recommendation
        total_cost = sum(a['cost'] for a in assignments)
        budget_utilization = total_cost / self.monthly_budget
        
        if budget_utilization > 0.9:
            recommendations.append(
                "Budget utilization is high (>90%). Consider reducing shots or using more simulators."
            )
        elif budget_utilization < 0.3:
            recommendations.append(
                "Budget utilization is low (<30%). You have capacity for more experiments."
            )
        
        return recommendations
    
    def forecast_costs(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cost forecast for experiments."""
        optimization_result = self.optimize_experiments(experiments)
        
        monthly_cost = optimization_result.optimized_cost
        budget_impact = monthly_cost / self.monthly_budget
        
        # Extrapolate to different time periods
        daily_cost = monthly_cost / 30
        quarterly_cost = monthly_cost * 3
        yearly_cost = monthly_cost * 12
        
        forecast = {
            'monthly_estimated_cost': monthly_cost,
            'daily_estimated_cost': daily_cost,
            'quarterly_estimated_cost': quarterly_cost,
            'yearly_estimated_cost': yearly_cost,
            'budget_impact': budget_impact,
            'experiments_count': len(experiments),
            'average_cost_per_experiment': monthly_cost / len(experiments) if experiments else 0,
            'cost_breakdown_by_provider': self._get_cost_breakdown_by_provider(optimization_result.optimized_assignments)
        }
        
        return forecast
    
    def _get_cost_breakdown_by_provider(self, assignments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get cost breakdown by provider."""
        provider_costs = {}
        
        for assignment in assignments:
            backend = assignment['backend']
            cost = assignment['cost']
            
            if backend in self.cost_models:
                provider = self.cost_models[backend].provider.value
                if provider not in provider_costs:
                    provider_costs[provider] = 0.0
                provider_costs[provider] += cost
        
        return provider_costs
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        total_spent = sum(quota.current_monthly_cost for quota in self.quotas.values())
        remaining_budget = self.monthly_budget - total_spent
        budget_utilization = total_spent / self.monthly_budget if self.monthly_budget > 0 else 0
        
        provider_status = {}
        for provider_type, quota in self.quotas.items():
            provider_status[provider_type.value] = {
                'monthly_spent': quota.current_monthly_cost,
                'monthly_limit': quota.monthly_cost_limit,
                'utilization': quota.current_monthly_cost / quota.monthly_cost_limit if quota.monthly_cost_limit > 0 else 0,
                'shots_used': quota.current_monthly_shots,
                'shots_limit': quota.monthly_shot_limit
            }
        
        return {
            'total_spent': total_spent,
            'monthly_budget': self.monthly_budget,
            'remaining_budget': remaining_budget,
            'budget_utilization': budget_utilization,
            'provider_breakdown': provider_status,
            'days_remaining_in_month': 30 - datetime.now().day,
            'daily_budget_remaining': remaining_budget / max(1, 30 - datetime.now().day)
        }
    
    def update_usage(self, provider: ProviderType, shots: int, cost: float):
        """Update usage tracking for a provider."""
        if provider in self.quotas:
            quota = self.quotas[provider]
            quota.current_monthly_shots += shots
            quota.current_monthly_cost += cost
            quota.current_daily_shots += shots
            quota.current_daily_cost += cost
            
            # Log usage history
            self.usage_history.append({
                'timestamp': datetime.now(),
                'provider': provider.value,
                'shots': shots,
                'cost': cost
            })
    
    def get_cost_recommendations(self) -> List[str]:
        """Get general cost optimization recommendations."""
        recommendations = []
        budget_status = self.get_budget_status()
        
        # Budget utilization recommendations
        if budget_status['budget_utilization'] > 0.9:
            recommendations.append(
                "⚠️ High budget utilization (>90%). Consider optimizing experiments or increasing budget."
            )
        
        # Provider distribution recommendations
        provider_costs = budget_status['provider_breakdown']
        if len(provider_costs) == 1:
            recommendations.append(
                "💡 Consider diversifying across multiple providers for better cost optimization."
            )
        
        # Usage pattern recommendations
        if len(self.usage_history) > 10:
            recent_usage = self.usage_history[-10:]
            avg_cost_per_job = sum(entry['cost'] for entry in recent_usage) / len(recent_usage)
            
            if avg_cost_per_job > 50:
                recommendations.append(
                    "💰 Average job cost is high. Consider using simulators for development."
                )
        
        return recommendations


def main():
    """Main entry point for cost optimization CLI."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Quantum cost optimizer')
    parser.add_argument('--budget', type=float, required=True, help='Monthly budget in USD')
    parser.add_argument('--experiments', required=True, help='Experiments specification file (JSON)')
    parser.add_argument('--optimize', action='store_true', help='Run cost optimization')
    parser.add_argument('--forecast', action='store_true', help='Generate cost forecast')
    parser.add_argument('--status', action='store_true', help='Show budget status')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = CostOptimizer(monthly_budget=args.budget)
    
    if args.experiments:
        with open(args.experiments, 'r') as f:
            experiments = json.load(f)
    else:
        experiments = []
    
    if args.optimize and experiments:
        # Run optimization
        result = optimizer.optimize_experiments(experiments)
        
        print("Cost Optimization Results:")
        print(f"Original cost: ${result.original_cost:.2f}")
        print(f"Optimized cost: ${result.optimized_cost:.2f}")
        print(f"Savings: ${result.savings:.2f} ({result.savings_percentage:.1f}%)")
        print(f"Optimizations applied: {', '.join(result.optimizations_applied)}")
    
    if args.forecast and experiments:
        # Generate forecast
        forecast = optimizer.forecast_costs(experiments)
        
        print("\nCost Forecast:")
        print(f"Monthly estimate: ${forecast['monthly_estimated_cost']:.2f}")
        print(f"Quarterly forecast: ${forecast['quarterly_forecast']:.2f}")
        print(f"Budget impact: {forecast['budget_impact']:.1%}")
        
        if forecast.get('recommendations'):
            print("Recommendations:")
            for rec in forecast['recommendations']:
                print(f"  • {rec}")
    
    if args.status:
        # Show budget status
        status = optimizer.get_budget_status()
        
        print("\nBudget Status:")
        print(f"Monthly budget: ${status['monthly_budget']:.2f}")
        print(f"Total spent: ${status['total_spent']:.2f}")
        print(f"Remaining: ${status['remaining_budget']:.2f}")
        print(f"Utilization: {status['budget_utilization']:.1%}")


if __name__ == '__main__':
    main()