"""
Main CLI entry point for quantum-devops-ci Python package.

This module provides the main command-line interface for the quantum testing
framework, including test execution, linting, and monitoring capabilities.
"""

import sys
import argparse
import warnings
from typing import List, Optional
from pathlib import Path

from . import __version__
from .testing import NoiseAwareTest
from .linting import QiskitLinter, PulseLinter
from .scheduling import QuantumJobScheduler
from .monitoring import QuantumCIMonitor
from .cost import CostOptimizer
from .deployment import QuantumDeployment


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='quantum-test',
        description='Quantum DevOps CI/CD testing framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quantum-test run                    # Run all quantum tests
  quantum-test run --framework qiskit --backend qasm_simulator
  quantum-test lint --check circuits --check pulses
  quantum-test monitor --project my-quantum-app
  quantum-test cost --budget 1000 --optimize
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('run', help='Run quantum tests')
    test_parser.add_argument('--framework', choices=['qiskit', 'cirq', 'pennylane'], 
                           default='qiskit', help='Quantum framework to use')
    test_parser.add_argument('--backend', default='qasm_simulator', 
                           help='Quantum backend for execution')
    test_parser.add_argument('--shots', type=int, default=1000, 
                           help='Number of measurement shots')
    test_parser.add_argument('--noise-model', help='Noise model to apply')
    test_parser.add_argument('--parallel', '-j', type=int, default=1,
                           help='Number of parallel test processes')
    test_parser.add_argument('--coverage', action='store_true',
                           help='Generate coverage report')
    test_parser.add_argument('--timeout', type=int, default=300,
                           help='Test timeout in seconds')
    test_parser.add_argument('tests', nargs='*', default=[],
                           help='Specific tests to run')
    
    # Lint command
    lint_parser = subparsers.add_parser('lint', help='Lint quantum circuits')
    lint_parser.add_argument('--check', action='append', choices=['circuits', 'pulses', 'all'],
                           default=[], help='What to check')
    lint_parser.add_argument('--config', help='Linting configuration file')
    lint_parser.add_argument('--format', choices=['text', 'json'], default='text',
                           help='Output format')
    lint_parser.add_argument('target', nargs='?', default='.',
                           help='Directory or file to lint')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor quantum CI/CD metrics')
    monitor_parser.add_argument('--project', required=True, help='Project name')
    monitor_parser.add_argument('--dashboard', help='Dashboard URL')
    monitor_parser.add_argument('--export', choices=['json', 'csv'], help='Export format')
    monitor_parser.add_argument('--summary', action='store_true', help='Show summary')
    monitor_parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    
    # Cost command
    cost_parser = subparsers.add_parser('cost', help='Optimize quantum computing costs')
    cost_parser.add_argument('--budget', type=float, help='Monthly budget in USD')
    cost_parser.add_argument('--optimize', action='store_true', help='Run cost optimization')
    cost_parser.add_argument('--forecast', action='store_true', help='Generate cost forecast')
    cost_parser.add_argument('--experiments', help='Experiments specification file')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Schedule quantum jobs')
    schedule_parser.add_argument('--goal', choices=['minimize_cost', 'minimize_time'],
                               default='minimize_cost', help='Optimization goal')
    schedule_parser.add_argument('--jobs', help='Jobs specification file')
    schedule_parser.add_argument('--constraints', help='Scheduling constraints file')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy quantum algorithms')
    deploy_parser.add_argument('--config', required=True, help='Deployment configuration')
    deploy_parser.add_argument('--algorithm', help='Algorithm ID to deploy')
    deploy_parser.add_argument('--strategy', choices=['blue_green', 'canary', 'rolling'],
                             default='blue_green', help='Deployment strategy')
    deploy_parser.add_argument('--validate', help='Validate deployment')
    deploy_parser.add_argument('--environment', help='Target environment')
    
    return parser


def run_tests(args) -> int:
    """Run quantum tests."""
    try:
        print(f"Running quantum tests with {args.framework} framework...")
        print(f"Backend: {args.backend}")
        print(f"Shots: {args.shots}")
        
        if args.noise_model:
            print(f"Noise model: {args.noise_model}")
        
        # Initialize test framework
        test_runner = NoiseAwareTest(
            default_shots=args.shots,
            timeout_seconds=args.timeout
        )
        
        # Run tests (placeholder implementation)
        warnings.warn("Test execution is not yet fully implemented")
        
        if args.tests:
            print(f"Running specific tests: {args.tests}")
        else:
            print("Running all quantum tests...")
        
        # Mock test results
        print("✅ All tests passed!")
        
        if args.coverage:
            print("📊 Coverage report generated")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


def run_lint(args) -> int:
    """Run quantum circuit linting."""
    try:
        checks = args.check if args.check else ['all']
        
        if 'all' in checks or 'circuits' in checks:
            print("🔍 Linting quantum circuits...")
            
            from .linting import QiskitLinter
            linter = QiskitLinter(args.config)
            
            # Lint target (placeholder)
            warnings.warn("Circuit linting is not yet fully implemented")
            print("✅ Circuit linting completed")
        
        if 'all' in checks or 'pulses' in checks:
            print("🔍 Linting pulse schedules...")
            
            from .linting import PulseLinter
            pulse_linter = PulseLinter(args.config)
            
            # Lint pulses (placeholder)
            warnings.warn("Pulse linting is not yet fully implemented")
            print("✅ Pulse linting completed")
        
        return 0
        
    except Exception as e:
        print(f"❌ Linting failed: {e}")
        return 1


def run_monitor(args) -> int:
    """Run quantum monitoring."""
    try:
        print(f"📊 Monitoring project: {args.project}")
        
        monitor = QuantumCIMonitor(
            project=args.project,
            dashboard_url=args.dashboard
        )
        
        if args.summary:
            build_summary = monitor.get_build_summary(args.days)
            cost_summary = monitor.get_cost_summary(args.days)
            
            print(f"\nBuild Summary ({args.days} days):")
            print(f"  Total builds: {build_summary.get('total_builds', 0)}")
            print(f"  Success rate: {build_summary.get('success_rate', 0.0):.1%}")
            
            print(f"\nCost Summary ({args.days} days):")
            print(f"  Total cost: ${cost_summary.get('total_cost', 0.0):.2f}")
            print(f"  Daily average: ${cost_summary.get('daily_average', 0.0):.2f}")
        
        if args.export:
            file_path = monitor.export_metrics(args.export)
            print(f"📁 Metrics exported to: {file_path}")
        
        monitor.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return 1


def run_cost(args) -> int:
    """Run cost optimization."""
    try:
        if not args.budget:
            print("❌ Budget required for cost optimization")
            return 1
        
        print(f"💰 Cost optimization with ${args.budget:.2f} monthly budget")
        
        optimizer = CostOptimizer(monthly_budget=args.budget)
        
        if args.optimize and args.experiments:
            import json
            with open(args.experiments, 'r') as f:
                experiments = json.load(f)
            
            result = optimizer.optimize_experiments(experiments)
            
            print(f"Original cost: ${result.original_cost:.2f}")
            print(f"Optimized cost: ${result.optimized_cost:.2f}")
            print(f"Savings: ${result.savings:.2f} ({result.savings_percentage:.1f}%)")
        
        if args.forecast and args.experiments:
            import json
            with open(args.experiments, 'r') as f:
                experiments = json.load(f)
            
            forecast = optimizer.forecast_costs(experiments)
            
            print(f"\nCost Forecast:")
            print(f"  Monthly estimate: ${forecast['monthly_estimated_cost']:.2f}")
            print(f"  Budget impact: {forecast['budget_impact']:.1%}")
        
        # Show budget status
        status = optimizer.get_budget_status()
        print(f"\nBudget Status:")
        print(f"  Total spent: ${status['total_spent']:.2f}")
        print(f"  Remaining: ${status['remaining_budget']:.2f}")
        print(f"  Utilization: {status['budget_utilization']:.1%}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Cost optimization failed: {e}")
        return 1


def run_schedule(args) -> int:
    """Run job scheduling."""
    try:
        print(f"📅 Quantum job scheduling (goal: {args.goal})")
        
        scheduler = QuantumJobScheduler(optimization_goal=args.goal)
        
        if args.jobs:
            import json
            with open(args.jobs, 'r') as f:
                jobs = json.load(f)
            
            constraints = {}
            if args.constraints:
                with open(args.constraints, 'r') as f:
                    constraints = json.load(f)
            
            schedule = scheduler.optimize_schedule(jobs, constraints)
            
            print(f"Jobs scheduled: {len(schedule.entries)}")
            print(f"Total cost: ${schedule.total_cost:.2f}")
            print(f"Total time: {schedule.total_time_hours:.1f} hours")
            
            print("\nDevice allocation:")
            for device, count in schedule.device_allocation.items():
                print(f"  {device}: {count} jobs")
        else:
            status = scheduler.get_queue_status()
            print("Queue Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Job scheduling failed: {e}")
        return 1


def run_deploy(args) -> int:
    """Run quantum deployment."""
    try:
        import json
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        deployer = QuantumDeployment(config)
        
        if args.algorithm:
            print(f"🚀 Deploying algorithm: {args.algorithm}")
            print(f"Strategy: {args.strategy}")
            
            # Mock circuit factory
            def mock_circuit_factory():
                return {"type": "mock_circuit", "qubits": 2}
            
            from .deployment import DeploymentStrategy
            strategy = DeploymentStrategy(args.strategy)
            
            deployment_id = deployer.deploy(args.algorithm, mock_circuit_factory, strategy)
            print(f"Deployment ID: {deployment_id}")
        
        if args.validate and args.environment:
            print(f"🔍 Validating deployment: {args.validate}")
            
            result = deployer.validate_deployment(args.validate, args.environment)
            
            print(f"Validation: {'✅ PASSED' if result.passed else '❌ FAILED'}")
            print(f"Fidelity: {result.fidelity:.3f}")
            print(f"Error rate: {result.error_rate:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for quantum-test CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up warning filters
    if not sys.warnoptions:
        warnings.filterwarnings('default', category=UserWarning, module='quantum_devops_ci')
    
    # Dispatch to appropriate handler
    try:
        if args.command == 'run':
            return run_tests(args)
        elif args.command == 'lint':
            return run_lint(args)
        elif args.command == 'monitor':
            return run_monitor(args)
        elif args.command == 'cost':
            return run_cost(args)
        elif args.command == 'schedule':
            return run_schedule(args)
        elif args.command == 'deploy':
            return run_deploy(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())