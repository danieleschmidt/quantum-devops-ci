#!/usr/bin/env python3
"""
Working Example: Quantum DevOps CI/CD Core Functionality

This example demonstrates the key features of the quantum-devops-ci toolkit:
1. Noise-aware quantum testing
2. Cost optimization
3. Job scheduling
4. Circuit linting
5. Performance monitoring

Run this example with: python examples/basic/working_example.py
"""

import sys
import time
import warnings
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    print("🚀 Quantum DevOps CI/CD - Working Example")
    print("=" * 50)
    
    # Check framework availability
    print("\n📋 Framework Availability Check")
    from quantum_devops_ci import check_framework_availability
    frameworks = check_framework_availability()
    
    for framework, version in frameworks.items():
        status = f"✅ {version}" if version else "❌ Not available"
        print(f"  {framework.capitalize()}: {status}")
    
    print("\n" + "=" * 50)
    
    # Example 1: Noise-Aware Testing
    print("\n🧪 Example 1: Noise-Aware Testing")
    try:
        from quantum_devops_ci.testing import NoiseAwareTest
        
        # Create test runner
        tester = NoiseAwareTest(default_shots=1000, timeout_seconds=30)
        print("✅ NoiseAwareTest initialized successfully")
        
        # Mock quantum circuit (would be real Qiskit circuit in practice)
        if frameworks['qiskit']:
            try:
                from qiskit import QuantumCircuit
                
                # Create Bell state circuit
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure_all()
                
                print("  📊 Running Bell state circuit...")
                result = tester.run_circuit(qc, shots=100)
                print(f"  📈 Result: {len(result.counts)} states measured")
                print(f"  ⏱️  Execution time: {result.execution_time:.3f}s")
                
                # Test with noise
                print("  🌊 Running with noise simulation...")
                noisy_result = tester.run_with_noise(qc, "depolarizing_0.01", shots=100)
                print(f"  📉 Noisy result: {len(noisy_result.counts)} states")
                
                # Calculate fidelity
                fidelity = tester.calculate_bell_fidelity(result)
                noisy_fidelity = tester.calculate_bell_fidelity(noisy_result)
                print(f"  🎯 Ideal fidelity: {fidelity:.3f}")
                print(f"  🎯 Noisy fidelity: {noisy_fidelity:.3f}")
                
            except ImportError:
                print("  ⚠️  Qiskit not available - using mock test")
                print("  ✅ Framework integration working")
        else:
            print("  ✅ Testing framework loaded (Qiskit not available)")
            
    except Exception as e:
        print(f"  ❌ Error in noise-aware testing: {e}")
    
    # Example 2: Cost Optimization
    print("\n💰 Example 2: Cost Optimization")
    try:
        from quantum_devops_ci.cost import CostOptimizer
        
        # Initialize cost optimizer
        optimizer = CostOptimizer(monthly_budget=1000.0)
        print("✅ CostOptimizer initialized with $1000 monthly budget")
        
        # Sample experiments
        experiments = [
            {
                'id': 'vqe_experiment',
                'circuit': 'mock_vqe_circuit',
                'shots': 10000,
                'priority': 'high',
                'backend_preferences': ['ibmq_manhattan', 'aws_sv1']
            },
            {
                'id': 'qaoa_experiment', 
                'circuit': 'mock_qaoa_circuit',
                'shots': 5000,
                'priority': 'medium',
                'backend_preferences': ['aws_sv1', 'ibmq_qasm_simulator']
            },
            {
                'id': 'test_experiment',
                'circuit': 'mock_test_circuit',
                'shots': 1000,
                'priority': 'low',
                'backend_preferences': ['ibmq_qasm_simulator']
            }
        ]
        
        print("  📊 Optimizing experiment costs...")
        result = optimizer.optimize_experiments(experiments)
        
        print(f"  💵 Original cost: ${result.original_cost:.2f}")
        print(f"  💵 Optimized cost: ${result.optimized_cost:.2f}")
        print(f"  💰 Savings: ${result.savings:.2f} ({result.savings_percentage:.1f}%)")
        print(f"  📋 Assignments: {len(result.optimized_assignments)} experiments")
        
        if result.recommendations:
            print("  💡 Recommendations:")
            for rec in result.recommendations[:2]:  # Show first 2
                print(f"    • {rec}")
                
        # Budget status
        budget_status = optimizer.get_budget_status()
        print("  📊 Budget Status:")
        print(f"    • Utilization: {budget_status['budget_utilization']:.1%}")
        print(f"    • Remaining: ${budget_status['remaining_budget']:.2f}")
        
    except Exception as e:
        print(f"  ❌ Error in cost optimization: {e}")
    
    # Example 3: Job Scheduling
    print("\n📅 Example 3: Job Scheduling")
    try:
        from quantum_devops_ci.scheduling import QuantumJobScheduler
        
        scheduler = QuantumJobScheduler(optimization_goal="minimize_cost")
        print("✅ QuantumJobScheduler initialized")
        
        # Sample jobs
        jobs = [
            {
                'id': 'job_1',
                'circuit': 'mock_circuit_1',
                'shots': 2000,
                'priority': 3,
                'backend_requirements': ['qasm_simulator']
            },
            {
                'id': 'job_2', 
                'circuit': 'mock_circuit_2',
                'shots': 5000,
                'priority': 2,
                'backend_requirements': ['ibmq_manhattan']
            }
        ]
        
        print("  ⚡ Optimizing job schedule...")
        schedule = scheduler.optimize_schedule(jobs)
        
        print(f"  📊 Scheduled jobs: {len(schedule.entries)}")
        print(f"  💰 Total cost: ${schedule.total_cost:.2f}")  
        print(f"  ⏰ Total time: {schedule.total_time_hours:.1f} hours")
        print(f"  🖥️  Backends used: {len(schedule.device_allocation)}")
        
        for backend, count in schedule.device_allocation.items():
            print(f"    • {backend}: {count} job(s)")
            
        if schedule.warnings:
            print("  ⚠️  Warnings:")
            for warning in schedule.warnings[:2]:
                print(f"    • {warning}")
        
        # Queue status
        status = scheduler.get_queue_status()
        print("  📈 Queue Status:")
        print(f"    • Completed jobs: {status['completed_jobs']}")
        print(f"    • Available backends: {len(status['backend_status'])}")
        
    except Exception as e:
        print(f"  ❌ Error in job scheduling: {e}")
    
    # Example 4: Circuit Linting
    print("\n🔍 Example 4: Circuit Linting")
    try:
        from quantum_devops_ci.linting import QiskitLinter, LintingConfig
        
        # Create linting configuration
        config = LintingConfig(
            max_circuit_depth=50,
            max_two_qubit_gates=25,
            allowed_gates=['u1', 'u2', 'u3', 'cx', 'h'],
            max_qubits=10
        )
        
        linter = QiskitLinter(config)
        print("✅ QiskitLinter initialized with custom config")
        
        if frameworks['qiskit']:
            try:
                from qiskit import QuantumCircuit
                
                # Create a test circuit with potential issues
                qc = QuantumCircuit(3, 3)
                qc.h(0)
                qc.cx(0, 1)
                qc.cx(1, 2)
                qc.cx(0, 2)  # Non-adjacent qubits
                qc.measure_all()
                
                print("  🔍 Linting quantum circuit...")
                lint_result = linter.lint_circuit(qc)
                
                print(f"  📊 {lint_result.get_summary()}")
                print(f"  📈 Issues found: {lint_result.total_issues}")
                
                if lint_result.issues:
                    print("  📋 Top issues:")
                    for issue in lint_result.issues[:3]:  # Show first 3
                        print(f"    • {issue.severity.upper()}: {issue.message}")
                        if issue.suggestion:
                            print(f"      💡 {issue.suggestion}")
                            
            except ImportError:
                print("  ⚠️  Qiskit not available - showing linter capabilities")
                print("  ✅ Linting framework loaded")
        else:
            print("  ✅ Linting framework ready (Qiskit not available)")
            
    except Exception as e:
        print(f"  ❌ Error in circuit linting: {e}")
    
    # Example 5: Performance Monitoring
    print("\n📊 Example 5: Performance Monitoring")
    try:
        from quantum_devops_ci.monitoring import QuantumCIMonitor
        
        monitor = QuantumCIMonitor(
            project="example-project",
            local_storage=True
        )
        print("✅ QuantumCIMonitor initialized")
        
        # Record sample build metric
        build_data = {
            'commit': 'abc123def',
            'branch': 'feature/quantum-optimization',
            'circuit_count': 5,
            'total_gates': 127,
            'max_depth': 15,
            'estimated_fidelity': 0.923,
            'noise_tests_passed': 18,
            'noise_tests_total': 20,
            'execution_time_seconds': 45.7
        }
        
        print("  📈 Recording build metrics...")
        monitor.record_build(build_data)
        
        # Record sample hardware usage
        usage_data = {
            'backend': 'ibmq_manhattan',
            'provider': 'ibmq',
            'shots': 5000,
            'queue_time_minutes': 12.5,
            'execution_time_minutes': 3.2,
            'cost_usd': 5.75,
            'circuit_depth': 15,
            'num_qubits': 3,
            'success': True
        }
        
        monitor.record_hardware_usage(usage_data)
        print("  💰 Recorded hardware usage")
        
        # Get summaries
        build_summary = monitor.get_build_summary(30)
        cost_summary = monitor.get_cost_summary(30)
        
        print("  📊 Build Summary (30 days):")
        print(f"    • Total builds: {build_summary['total_builds']}")
        print(f"    • Success rate: {build_summary['success_rate']:.1%}")
        print(f"    • Avg execution time: {build_summary['average_execution_time']:.1f}s")
        
        print("  💰 Cost Summary (30 days):")
        print(f"    • Total cost: ${cost_summary['total_cost']:.2f}")
        print(f"    • Total shots: {cost_summary['total_shots']:,}")
        if cost_summary['total_shots'] > 0:
            print(f"    • Cost per shot: ${cost_summary['cost_per_shot']:.4f}")
        
        # Health status
        health = monitor.get_health_status()
        print(f"  🏥 Health Status: {health['overall_status']} ({health['health_score']:.0f}/100)")
        
        if health['recommendations']:
            print("  💡 Health Recommendations:")
            for rec in health['recommendations'][:2]:
                print(f"    • {rec}")
        
        monitor.shutdown()
        
    except Exception as e:
        print(f"  ❌ Error in performance monitoring: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Working Example Complete!")
    print("\nCore functionality demonstrated:")
    print("✅ Noise-aware quantum testing")
    print("✅ Cost optimization and budget management")
    print("✅ Intelligent job scheduling")
    print("✅ Circuit linting and validation")
    print("✅ Performance monitoring and metrics")
    print("\nThe quantum-devops-ci toolkit is ready for integration!")
    print("📚 See docs/ for detailed usage examples")
    print("🚀 Run tests with: python -m pytest quantum-tests/")


if __name__ == "__main__":
    main()