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
    reset_date: datetime = field(default_factory=lambda: datetime.now().replace(day=1))


@dataclass 
class CostEstimate:
    """Cost estimation for quantum job or batch."""
    total_cost: float
    breakdown_by_provider: Dict[str, float]
    breakdown_by_backend: Dict[str, float]
    estimated_savings: float = 0.0
    optimization_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of cost optimization."""
    original_cost: float
    optimized_cost: float
    savings: float
    savings_percentage: float
    job_assignments: Dict[str, str]  # job_id -> backend_name
    optimizations_applied: List[str]
    
    def __post_init__(self):
        if self.original_cost > 0:
            self.savings_percentage = (self.savings / self.original_cost) * 100
        else:
            self.savings_percentage = 0.0


class CostOptimizer:
    """
    Cost optimizer for quantum computing workloads.
    
    This optimizer analyzes quantum computing costs across providers and
    suggests optimal resource allocation to minimize expenses while meeting
    performance and deadline requirements.
    """
    
    def __init__(
        self,
        monthly_budget: float,
        priority_weights: Optional[Dict[str, float]] = None,
        provider_preferences: Optional[List[ProviderType]] = None
    ):
        """
        Initialize cost optimizer.
        
        Args:
            monthly_budget: Monthly budget in USD
            priority_weights: Weights for different priority levels
            provider_preferences: Preferred providers in order
        """
        self.monthly_budget = monthly_budget
        self.priority_weights = priority_weights or {
            "critical": 0.4,
            "production": 0.3,
            "research": 0.2,
            "development": 0.1
        }
        self.provider_preferences = provider_preferences or []
        
        # Initialize cost models and quotas
        self.cost_models: Dict[str, CostModel] = {}
        self.usage_quotas: Dict[ProviderType, UsageQuota] = {}
        
        # Load default cost models
        self._initialize_cost_models()
        self._initialize_usage_quotas()
    
    def estimate_cost(
        self,
        jobs: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> CostEstimate:
        """
        Estimate cost for batch of quantum jobs.
        
        Args:
            jobs: List of job specifications
            constraints: Optional constraints (deadline, provider preferences)
            
        Returns:
            Cost estimation with breakdown
        """
        if constraints is None:
            constraints = {}
        
        total_cost = 0.0
        provider_costs = {}
        backend_costs = {}
        warnings_list = []
        
        for job in jobs:
            shots = job.get("shots", 1000)
            priority = job.get("priority", "research")
            backend_preference = job.get("backend")
            
            # Find best backend for this job
            if backend_preference:
                backend_name = backend_preference
                if backend_name not in self.cost_models:
                    warnings_list.append(f"Unknown backend: {backend_name}")
                    continue
                cost_model = self.cost_models[backend_name]
            else:
                cost_model = self._select_cheapest_backend(shots, priority, constraints)
                backend_name = cost_model.backend_name
            
            # Calculate job cost
            job_cost = self._calculate_job_cost(shots, cost_model, priority)
            
            total_cost += job_cost
            
            # Update breakdowns
            provider_name = cost_model.provider.value
            provider_costs[provider_name] = provider_costs.get(provider_name, 0.0) + job_cost
            backend_costs[backend_name] = backend_costs.get(backend_name, 0.0) + job_cost
        
        return CostEstimate(
            total_cost=total_cost,
            breakdown_by_provider=provider_costs,
            breakdown_by_backend=backend_costs,
            warnings=warnings_list
        )
    
    def optimize_experiments(
        self,
        experiments: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize experiment batch for cost efficiency.
        
        Args:
            experiments: List of experiment specifications
            constraints: Optimization constraints
            
        Returns:
            Optimization result with savings analysis
        """
        if constraints is None:
            constraints = {}
        
        # Calculate original cost (naive assignment)
        original_estimate = self.estimate_cost(experiments, constraints)
        original_cost = original_estimate.total_cost
        
        # Apply optimization strategies
        optimized_assignments = {}
        optimizations_applied = []
        
        # Strategy 1: Batch similar jobs for bulk discounts
        if self._can_apply_bulk_optimization(experiments):
            optimized_assignments.update(self._optimize_bulk_jobs(experiments))
            optimizations_applied.append("bulk_discounts")
        
        # Strategy 2: Use cheaper backends where possible
        provider_optimized = self._optimize_provider_selection(experiments, constraints)
        optimized_assignments.update(provider_optimized)
        optimizations_applied.append("provider_optimization")
        
        # Strategy 3: Leverage off-peak pricing (if available)
        if constraints.get("flexible_timing", False):
            time_optimized = self._optimize_timing(experiments)
            optimized_assignments.update(time_optimized)
            optimizations_applied.append("time_optimization")
        
        # Calculate optimized cost
        optimized_cost = self._calculate_optimized_cost(experiments, optimized_assignments)
        savings = original_cost - optimized_cost
        
        return OptimizationResult(
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            savings=savings,
            job_assignments=optimized_assignments,
            optimizations_applied=optimizations_applied
        )
    
    def track_usage(self, provider: ProviderType, shots: int, cost: float):
        """
        Track usage against quotas.
        
        Args:
            provider: Quantum provider
            shots: Number of shots used
            cost: Cost incurred
        """
        if provider not in self.usage_quotas:
            warnings.warn(f"No quota configured for provider: {provider}")
            return
        
        quota = self.usage_quotas[provider]
        
        # Update usage
        quota.current_monthly_shots += shots
        quota.current_monthly_cost += cost
        quota.current_daily_shots += shots
        quota.current_daily_cost += cost
        
        # Check limits
        if quota.current_monthly_cost > quota.monthly_cost_limit:
            warnings.warn(f"Monthly cost limit exceeded for {provider.value}: "
                         f"${quota.current_monthly_cost:.2f} > ${quota.monthly_cost_limit:.2f}")
        
        if quota.daily_cost_limit and quota.current_daily_cost > quota.daily_cost_limit:
            warnings.warn(f"Daily cost limit exceeded for {provider.value}: "
                         f"${quota.current_daily_cost:.2f} > ${quota.daily_cost_limit:.2f}")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """
        Get current budget and usage status.
        
        Returns:
            Budget status information
        """
        total_monthly_cost = sum(q.current_monthly_cost for q in self.usage_quotas.values())
        remaining_budget = self.monthly_budget - total_monthly_cost
        
        provider_usage = {}
        for provider, quota in self.usage_quotas.items():
            provider_usage[provider.value] = {
                "monthly_cost": quota.current_monthly_cost,
                "monthly_shots": quota.current_monthly_shots,
                "cost_limit": quota.monthly_cost_limit,
                "shot_limit": quota.monthly_shot_limit,
                "utilization": quota.current_monthly_cost / quota.monthly_cost_limit if quota.monthly_cost_limit > 0 else 0
            }
        
        return {
            "monthly_budget": self.monthly_budget,
            "total_spent": total_monthly_cost,
            "remaining_budget": remaining_budget,
            "budget_utilization": total_monthly_cost / self.monthly_budget if self.monthly_budget > 0 else 0,
            "provider_usage": provider_usage
        }
    
    def suggest_cost_savings(self, usage_history: List[Dict[str, Any]]) -> List[str]:
        """
        Suggest cost-saving opportunities based on usage history.
        
        Args:
            usage_history: Historical usage data
            
        Returns:
            List of cost-saving suggestions
        """
        suggestions = []
        
        if not usage_history:
            return suggestions
        
        # Analyze usage patterns
        total_cost = sum(u.get("cost", 0) for u in usage_history)
        provider_costs = {}
        backend_usage = {}
        
        for usage in usage_history:
            provider = usage.get("provider", "unknown")
            backend = usage.get("backend", "unknown")
            cost = usage.get("cost", 0)
            
            provider_costs[provider] = provider_costs.get(provider, 0) + cost
            backend_usage[backend] = backend_usage.get(backend, 0) + 1
        
        # Suggest bulk optimization
        if any(count > 50 for count in backend_usage.values()):
            suggestions.append("Consider batching similar jobs to leverage bulk discounts")
        
        # Suggest provider diversification
        if len(provider_costs) == 1:
            suggestions.append("Consider using multiple providers to optimize costs and reduce vendor lock-in")
        
        # Suggest simulator usage for development
        if any("simulator" not in backend for backend in backend_usage.keys()):
            suggestions.append("Use simulators for development and testing to reduce hardware costs")
        
        # Budget optimization
        if total_cost > self.monthly_budget * 0.8:
            suggestions.append("Current usage is approaching budget limits - consider cost optimization")
        
        return suggestions
    
    def forecast_costs(
        self,
        planned_experiments: List[Dict[str, Any]],
        months: int = 3
    ) -> Dict[str, Any]:
        """
        Forecast costs for planned experiments.
        
        Args:
            planned_experiments: List of planned experiment specifications
            months: Number of months to forecast
            
        Returns:
            Cost forecast analysis
        """
        monthly_cost = self.estimate_cost(planned_experiments).total_cost
        
        forecast = {
            "monthly_estimated_cost": monthly_cost,
            "quarterly_forecast": monthly_cost * months,
            "annual_forecast": monthly_cost * 12,
            "budget_impact": monthly_cost / self.monthly_budget if self.monthly_budget > 0 else 0
        }
        
        # Add recommendations
        if monthly_cost > self.monthly_budget:
            forecast["recommendations"] = [
                "Estimated costs exceed monthly budget",
                "Consider optimizing experiment parameters",
                "Use simulators where possible",
                "Implement staged execution"
            ]
        elif monthly_cost > self.monthly_budget * 0.8:
            forecast["recommendations"] = [
                "Costs approaching budget limits",
                "Monitor usage closely",
                "Consider cost optimization strategies"
            ]
        else:
            forecast["recommendations"] = [
                "Costs within budget",
                "Good opportunity for additional experiments"
            ]
        
        return forecast
    
    def _initialize_cost_models(self):
        """Initialize default cost models for quantum providers."""
        # IBM Quantum
        self.cost_models["ibmq_qasm_simulator"] = CostModel(
            provider=ProviderType.IBMQ,
            backend_name="ibmq_qasm_simulator",
            cost_per_shot=0.0001,
            bulk_discount_threshold=10000,
            bulk_discount_rate=0.1
        )
        
        self.cost_models["ibmq_manhattan"] = CostModel(
            provider=ProviderType.IBMQ,
            backend_name="ibmq_manhattan",
            cost_per_shot=0.001,
            minimum_cost=0.10,
            setup_cost=0.05
        )
        
        # AWS Braket
        self.cost_models["aws_sv1"] = CostModel(
            provider=ProviderType.AWS_BRAKET,
            backend_name="aws_sv1",
            cost_per_shot=0.00075,
            cost_per_minute=0.075
        )
        
        self.cost_models["rigetti_aspen"] = CostModel(
            provider=ProviderType.RIGETTI,
            backend_name="rigetti_aspen",
            cost_per_shot=0.0015,
            minimum_cost=0.30,
            priority_multiplier=1.5
        )
        
        # IonQ
        self.cost_models["ionq_qpu"] = CostModel(
            provider=ProviderType.IONQ,
            backend_name="ionq_qpu",
            cost_per_shot=0.002,
            minimum_cost=0.50,
            setup_cost=0.10
        )
    
    def _initialize_usage_quotas(self):
        """Initialize default usage quotas."""
        budget_per_provider = self.monthly_budget / 3  # Distribute across 3 main providers
        
        self.usage_quotas[ProviderType.IBMQ] = UsageQuota(
            provider=ProviderType.IBMQ,
            monthly_shot_limit=1000000,
            monthly_cost_limit=budget_per_provider,
            daily_cost_limit=budget_per_provider / 30
        )
        
        self.usage_quotas[ProviderType.AWS_BRAKET] = UsageQuota(
            provider=ProviderType.AWS_BRAKET,
            monthly_shot_limit=500000,
            monthly_cost_limit=budget_per_provider,
            daily_cost_limit=budget_per_provider / 30
        )
        
        self.usage_quotas[ProviderType.IONQ] = UsageQuota(
            provider=ProviderType.IONQ,
            monthly_shot_limit=100000,
            monthly_cost_limit=budget_per_provider,
            daily_cost_limit=budget_per_provider / 30
        )
    
    def _select_cheapest_backend(
        self,
        shots: int,
        priority: str,
        constraints: Dict[str, Any]
    ) -> CostModel:
        """Select the cheapest suitable backend."""
        suitable_backends = []
        
        for cost_model in self.cost_models.values():
            # Check provider preferences
            if self.provider_preferences and cost_model.provider not in self.provider_preferences:
                continue
            
            # Check quota availability
            quota = self.usage_quotas.get(cost_model.provider)
            if quota:
                estimated_cost = self._calculate_job_cost(shots, cost_model, priority)
                if quota.current_monthly_cost + estimated_cost > quota.monthly_cost_limit:
                    continue
            
            suitable_backends.append(cost_model)
        
        if not suitable_backends:
            # Fallback to first available backend
            return list(self.cost_models.values())[0]
        
        # Return cheapest option
        return min(suitable_backends, key=lambda cm: self._calculate_job_cost(shots, cm, priority))
    
    def _calculate_job_cost(self, shots: int, cost_model: CostModel, priority: str) -> float:
        """Calculate cost for a single job."""
        base_cost = shots * cost_model.cost_per_shot
        
        # Apply bulk discount
        if shots >= cost_model.bulk_discount_threshold:
            discount = base_cost * cost_model.bulk_discount_rate
            base_cost -= discount
        
        # Apply priority multiplier
        priority_weight = self.priority_weights.get(priority, 1.0)
        base_cost *= priority_weight * cost_model.priority_multiplier
        
        # Add fixed costs
        total_cost = base_cost + cost_model.setup_cost
        
        # Apply minimum cost
        return max(total_cost, cost_model.minimum_cost)
    
    def _can_apply_bulk_optimization(self, experiments: List[Dict[str, Any]]) -> bool:
        """Check if bulk optimization can be applied."""
        total_shots = sum(exp.get("shots", 1000) for exp in experiments)
        return total_shots >= 10000
    
    def _optimize_bulk_jobs(self, experiments: List[Dict[str, Any]]) -> Dict[str, str]:
        """Optimize jobs for bulk discounts."""
        assignments = {}
        
        # Group experiments by similarity (simplified)
        similar_groups = {}
        for i, exp in enumerate(experiments):
            shots = exp.get("shots", 1000)
            group_key = f"shots_{shots//1000}k"  # Group by shot ranges
            
            if group_key not in similar_groups:
                similar_groups[group_key] = []
            similar_groups[group_key].append(f"job_{i}")
        
        # Assign groups to cheapest backends with bulk discounts
        for group_jobs in similar_groups.values():
            backend = min(self.cost_models.values(),
                         key=lambda cm: cm.cost_per_shot * (1 - cm.bulk_discount_rate))
            
            for job_id in group_jobs:
                assignments[job_id] = backend.backend_name
        
        return assignments
    
    def _optimize_provider_selection(
        self,
        experiments: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, str]:
        """Optimize provider selection for cost efficiency."""
        assignments = {}
        
        for i, exp in enumerate(experiments):
            job_id = f"job_{i}"
            shots = exp.get("shots", 1000)
            priority = exp.get("priority", "research")
            
            # Find cheapest available backend
            best_backend = self._select_cheapest_backend(shots, priority, constraints)
            assignments[job_id] = best_backend.backend_name
        
        return assignments
    
    def _optimize_timing(self, experiments: List[Dict[str, Any]]) -> Dict[str, str]:
        """Optimize job timing for off-peak pricing."""
        assignments = {}
        
        # Placeholder implementation - would integrate with provider pricing APIs
        for i, exp in enumerate(experiments):
            job_id = f"job_{i}"
            # Assign to simulator for off-peak timing (placeholder logic)
            assignments[job_id] = "ibmq_qasm_simulator"
        
        return assignments
    
    def _calculate_optimized_cost(
        self,
        experiments: List[Dict[str, Any]],
        assignments: Dict[str, str]
    ) -> float:
        """Calculate total cost with optimized assignments."""
        total_cost = 0.0
        
        for i, exp in enumerate(experiments):
            job_id = f"job_{i}"
            backend_name = assignments.get(job_id, "ibmq_qasm_simulator")
            
            if backend_name in self.cost_models:
                cost_model = self.cost_models[backend_name]
                shots = exp.get("shots", 1000)
                priority = exp.get("priority", "research")
                
                job_cost = self._calculate_job_cost(shots, cost_model, priority)
                total_cost += job_cost
        
        return total_cost


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