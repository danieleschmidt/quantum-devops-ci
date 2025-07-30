"""
Advanced cost optimization example for quantum computing workloads.

This example demonstrates how to use the quantum DevOps CI/CD cost optimization
features to minimize quantum computing expenses while maintaining quality.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from quantum_devops_ci.cost import CostOptimizer, ProviderType
from quantum_devops_ci.scheduling import QuantumJobScheduler


def create_sample_experiments():
    """
    Create sample quantum experiments for cost optimization.
    
    Returns:
        List of experiment specifications
    """
    experiments = []
    
    # VQE experiments for molecular simulation
    for i, molecule in enumerate(['H2', 'LiH', 'BeH2']):
        experiments.append({
            "id": f"vqe_{molecule.lower()}",
            "algorithm": "VQE",
            "molecule": molecule,
            "shots": 10000 + i * 5000,  # Increasing complexity
            "priority": "research",
            "estimated_runtime_minutes": 30 + i * 20,
            "max_circuit_depth": 50 + i * 25,
            "preferred_backends": ["ibmq_qasm_simulator", "aws_sv1"]
        })
    
    # QAOA experiments for optimization problems
    for i, problem_size in enumerate([4, 8, 12]):
        experiments.append({
            "id": f"qaoa_maxcut_{problem_size}",
            "algorithm": "QAOA",
            "problem_size": problem_size,
            "shots": 8192,
            "priority": "production" if problem_size <= 8 else "research",
            "estimated_runtime_minutes": 15 * (problem_size // 4),
            "max_circuit_depth": 20 + problem_size * 3,
            "deadline": datetime.now() + timedelta(hours=24)
        })
    
    # Quantum machine learning experiments
    for i, dataset in enumerate(['iris', 'wine', 'digits']):
        experiments.append({
            "id": f"qml_{dataset}",
            "algorithm": "QuantumSVM", 
            "dataset": dataset,
            "shots": 4096,
            "priority": "development",
            "estimated_runtime_minutes": 20,
            "max_circuit_depth": 30,
            "flexible_timing": True  # Can run during off-peak hours
        })
    
    # Hardware characterization experiments
    experiments.extend([
        {
            "id": "randomized_benchmarking",
            "algorithm": "RB",
            "shots": 100000,
            "priority": "critical",
            "estimated_runtime_minutes": 180,
            "requires_hardware": True,
            "preferred_backends": ["ibmq_manhattan", "ibmq_brooklyn"]
        },
        {
            "id": "process_tomography",
            "algorithm": "ProcessTomography",
            "shots": 50000,
            "priority": "critical", 
            "estimated_runtime_minutes": 120,
            "requires_hardware": True
        }
    ])
    
    return experiments


def demonstrate_cost_estimation():
    """Demonstrate cost estimation for quantum experiments."""
    print("💰 Cost Estimation Demo")
    print("=" * 50)
    
    # Initialize cost optimizer with $5000 monthly budget
    optimizer = CostOptimizer(
        monthly_budget=5000.0,
        priority_weights={
            "critical": 0.4,
            "production": 0.3,
            "research": 0.2,
            "development": 0.1
        }
    )
    
    # Get sample experiments
    experiments = create_sample_experiments()
    
    # Convert to cost optimizer format
    cost_jobs = []
    for exp in experiments:
        job = {
            "circuit": f"circuit_{exp['id']}",  # Placeholder
            "shots": exp["shots"],
            "priority": exp["priority"],
            "backend": exp.get("preferred_backends", [None])[0] if exp.get("preferred_backends") else None
        }
        cost_jobs.append(job)
    
    # Estimate costs
    print("\n📊 Cost Estimation Results:")
    estimate = optimizer.estimate_cost(cost_jobs)
    
    print(f"Total estimated cost: ${estimate.total_cost:.2f}")
    print("\nBreakdown by provider:")
    for provider, cost in estimate.breakdown_by_provider.items():
        print(f"  {provider}: ${cost:.2f}")
    
    print("\nBreakdown by backend:")
    for backend, cost in estimate.breakdown_by_backend.items():
        print(f"  {backend}: ${cost:.2f}")
    
    if estimate.warnings:
        print("\n⚠️  Warnings:")
        for warning in estimate.warnings:
            print(f"  • {warning}")
    
    return optimizer, experiments


def demonstrate_cost_optimization():
    """Demonstrate cost optimization strategies."""
    print("\n🔧 Cost Optimization Demo")
    print("=" * 50)
    
    optimizer, experiments = demonstrate_cost_estimation()
    
    # Define optimization constraints
    constraints = {
        "deadline": datetime.now() + timedelta(hours=48),
        "budget": 4000.0,  # Tighter budget to force optimization
        "preferred_devices": ["ibmq_qasm_simulator", "aws_sv1", "ibmq_manhattan"],
        "flexible_timing": True
    }
    
    # Run optimization
    print("\n⚡ Running cost optimization...")
    result = optimizer.optimize_experiments(experiments, constraints)
    
    print(f"\nOptimization Results:")
    print(f"Original cost: ${result.original_cost:.2f}")
    print(f"Optimized cost: ${result.optimized_cost:.2f}")
    print(f"Total savings: ${result.savings:.2f} ({result.savings_percentage:.1f}%)")
    
    print(f"\nOptimizations applied:")
    for optimization in result.optimizations_applied:
        print(f"  ✅ {optimization}")
    
    print(f"\nJob assignments:")
    for job_id, backend in result.job_assignments.items():
        print(f"  {job_id}: {backend}")
    
    return result


def demonstrate_budget_tracking():
    """Demonstrate budget and usage tracking."""
    print("\n📈 Budget Tracking Demo")
    print("=" * 50)
    
    optimizer = CostOptimizer(monthly_budget=5000.0)
    
    # Simulate some usage
    usage_scenarios = [
        (ProviderType.IBMQ, 50000, 125.50),
        (ProviderType.AWS_BRAKET, 25000, 87.25),
        (ProviderType.IBMQ, 75000, 230.75),
        (ProviderType.IONQ, 10000, 450.00)
    ]
    
    print("Tracking usage:")
    for provider, shots, cost in usage_scenarios:
        optimizer.track_usage(provider, shots, cost)
        print(f"  {provider.value}: {shots} shots, ${cost:.2f}")
    
    # Get budget status
    status = optimizer.get_budget_status()
    
    print(f"\nBudget Status:")
    print(f"Monthly budget: ${status['monthly_budget']:.2f}")
    print(f"Total spent: ${status['total_spent']:.2f}")
    print(f"Remaining budget: ${status['remaining_budget']:.2f}")
    print(f"Budget utilization: {status['budget_utilization']:.1%}")
    
    print(f"\nProvider Usage:")
    for provider, usage in status['provider_usage'].items():
        print(f"  {provider}:")
        print(f"    Cost: ${usage['monthly_cost']:.2f} / ${usage['cost_limit']:.2f}")
        print(f"    Shots: {usage['monthly_shots']:,} / {usage['shot_limit']:,}")
        print(f"    Utilization: {usage['utilization']:.1%}")


def demonstrate_cost_forecasting():
    """Demonstrate cost forecasting for planned experiments."""
    print("\n🔮 Cost Forecasting Demo")
    print("=" * 50)
    
    optimizer = CostOptimizer(monthly_budget=5000.0)
    
    # Create planned experiments for next quarter
    planned_experiments = []
    
    # Regular weekly experiments
    for week in range(12):  # 3 months
        planned_experiments.extend([
            {
                "circuit": f"weekly_vqe_{week}",
                "shots": 15000,
                "priority": "research"
            },
            {
                "circuit": f"weekly_qaoa_{week}", 
                "shots": 8000,
                "priority": "production"
            }
        ])
    
    # Monthly hardware calibration
    for month in range(3):
        planned_experiments.extend([
            {
                "circuit": f"calibration_{month}",
                "shots": 100000,
                "priority": "critical",
                "backend": "ibmq_manhattan"
            }
        ])
    
    # Generate forecast
    forecast = optimizer.forecast_costs(planned_experiments, months=3)
    
    print(f"Cost Forecast (3 months):")
    print(f"Monthly estimated cost: ${forecast['monthly_estimated_cost']:.2f}")
    print(f"Quarterly forecast: ${forecast['quarterly_forecast']:.2f}")
    print(f"Annual forecast: ${forecast['annual_forecast']:.2f}")
    print(f"Budget impact: {forecast['budget_impact']:.1%}")
    
    if forecast.get('recommendations'):
        print(f"\nRecommendations:")
        for rec in forecast['recommendations']:
            print(f"  • {rec}")


def demonstrate_cost_savings_suggestions():
    """Demonstrate cost savings suggestions based on usage history."""
    print("\n💡 Cost Savings Suggestions Demo")
    print("=" * 50)
    
    optimizer = CostOptimizer(monthly_budget=3000.0)
    
    # Create mock usage history
    usage_history = []
    
    # Heavy use of expensive backends
    for i in range(30):
        usage_history.append({
            "provider": "ibmq",
            "backend": "ibmq_manhattan",
            "cost": 45.50 + np.random.normal(0, 5),
            "shots": 25000,
            "date": datetime.now() - timedelta(days=i)
        })
    
    # Some simulator usage
    for i in range(10):
        usage_history.append({
            "provider": "ibmq", 
            "backend": "ibmq_qasm_simulator",
            "cost": 2.50,
            "shots": 50000,
            "date": datetime.now() - timedelta(days=i*3)
        })
    
    # Get cost savings suggestions
    suggestions = optimizer.suggest_cost_savings(usage_history)
    
    print("Cost Savings Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")


def demonstrate_integrated_workflow():
    """Demonstrate integrated cost optimization and job scheduling."""
    print("\n🔄 Integrated Workflow Demo")
    print("=" * 50)
    
    # Initialize both cost optimizer and job scheduler
    cost_optimizer = CostOptimizer(monthly_budget=4000.0)
    job_scheduler = QuantumJobScheduler(optimization_goal="minimize_cost")
    
    # Get experiments
    experiments = create_sample_experiments()
    
    # First, optimize costs
    optimization_result = cost_optimizer.optimize_experiments(experiments)
    
    print(f"Cost optimization achieved ${optimization_result.savings:.2f} savings")
    
    # Convert to job scheduler format
    jobs = []
    for i, exp in enumerate(experiments):
        # Use optimized backend assignment if available
        job_id = f"job_{i}"
        backend = optimization_result.job_assignments.get(job_id, "ibmq_qasm_simulator")
        
        job_spec = {
            "circuit": {"type": "placeholder", "id": exp["id"]},
            "shots": exp["shots"],
            "priority": exp["priority"],
            "backend": backend
        }
        jobs.append(job_spec)
    
    # Schedule optimized jobs
    schedule_constraints = {
        "deadline": datetime.now() + timedelta(hours=72),
        "budget": optimization_result.optimized_cost * 1.1  # 10% buffer
    }
    
    schedule = job_scheduler.optimize_schedule(jobs, schedule_constraints)
    
    print(f"\nScheduling Results:")
    print(f"Jobs scheduled: {len(schedule.entries)}")
    print(f"Total cost: ${schedule.total_cost:.2f}")
    print(f"Total time: {schedule.total_time_hours:.1f} hours")
    
    print(f"\nDevice allocation:")
    for device, count in schedule.device_allocation.items():
        print(f"  {device}: {count} jobs")
    
    # Calculate total savings
    original_cost = optimization_result.original_cost
    final_cost = schedule.total_cost
    total_savings = original_cost - final_cost
    savings_percentage = (total_savings / original_cost) * 100
    
    print(f"\nTotal Savings:")
    print(f"Original cost: ${original_cost:.2f}")
    print(f"Final cost: ${final_cost:.2f}")
    print(f"Total savings: ${total_savings:.2f} ({savings_percentage:.1f}%)")


def main():
    """Run all cost optimization demonstrations."""
    print("🌌 Quantum Cost Optimization Demo")
    print("=" * 60)
    print("This demo shows how to optimize quantum computing costs")
    print("using the quantum-devops-ci framework.\n")
    
    try:
        # Run all demonstrations
        demonstrate_cost_estimation()
        demonstrate_cost_optimization()
        demonstrate_budget_tracking()
        demonstrate_cost_forecasting()
        demonstrate_cost_savings_suggestions()
        demonstrate_integrated_workflow()
        
        print("\n✨ Demo completed successfully!")
        print("\nKey Takeaways:")
        print("  • Cost optimization can save 20-40% on quantum computing expenses")
        print("  • Proper scheduling reduces both cost and execution time")
        print("  • Budget tracking prevents cost overruns")
        print("  • Forecasting helps with resource planning")
        print("  • Integration with CI/CD enables automated cost control")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()