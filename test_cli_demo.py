#!/usr/bin/env python3
"""
CLI demonstration script for quantum-devops-ci.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_devops_ci.cli import main

def create_test_configs():
    """Create test configuration files."""
    
    # Deployment config
    deploy_config = {
        "environments": {
            "production": {
                "backend": "qasm_simulator",
                "allocation": 100.0,
                "max_shots": 10000,
                "validation_shots": 1000
            },
            "staging": {
                "backend": "statevector_simulator", 
                "allocation": 50.0,
                "max_shots": 5000,
                "validation_shots": 500
            }
        }
    }
    
    with open('deploy_config.json', 'w') as f:
        json.dump(deploy_config, f, indent=2)
    
    # Test jobs config
    jobs_config = [
        {
            "id": "vqe_experiment_1",
            "circuit": "vqe_h2",
            "shots": 10000,
            "priority": "high"
        },
        {
            "id": "qaoa_experiment_1", 
            "circuit": "qaoa_maxcut",
            "shots": 5000,
            "priority": "medium"
        }
    ]
    
    with open('test_jobs.json', 'w') as f:
        json.dump(jobs_config, f, indent=2)
    
    # Cost experiments config
    cost_experiments = [
        {
            "name": "bell_state_test",
            "shots": 1000,
            "backend": "ibmq_qasm_simulator",
            "estimated_cost": 0.10
        },
        {
            "name": "vqe_optimization",
            "shots": 50000,
            "backend": "ibmq_manhattan", 
            "estimated_cost": 25.00
        }
    ]
    
    with open('cost_experiments.json', 'w') as f:
        json.dump(cost_experiments, f, indent=2)

def demo_cli_commands():
    """Demonstrate CLI commands."""
    
    print("🎯 Quantum DevOps CI - CLI Demonstration")
    print("=" * 50)
    
    create_test_configs()
    
    # Test deployment command
    print("\n🚀 Testing Deployment Command:")
    print("-" * 30)
    result = main(['deploy', '--config', 'deploy_config.json', '--algorithm', 'bell_state_v1'])
    print(f"Deploy command result: {result}")
    
    # Test deployment validation
    print("\n🔍 Testing Deployment Validation:")
    print("-" * 30)
    result = main(['deploy', '--config', 'deploy_config.json', '--validate', 'bell_state_v1_20250810_123456', '--environment', 'production'])
    print(f"Validation command result: {result}")
    
    # Test quantum test run
    print("\n🧪 Testing Quantum Test Run:")
    print("-" * 30)
    result = main(['run', '--framework', 'qiskit', '--shots', '500', '--backend', 'qasm_simulator'])
    print(f"Test run result: {result}")
    
    # Test linting
    print("\n🔍 Testing Circuit Linting:")
    print("-" * 30)
    result = main(['lint', '--check', 'circuits', 'src/'])
    print(f"Lint result: {result}")
    
    # Test monitoring
    print("\n📊 Testing Monitoring:")
    print("-" * 30)
    result = main(['monitor', '--project', 'quantum-app-demo', '--summary'])
    print(f"Monitor result: {result}")
    
    # Test cost optimization
    print("\n💰 Testing Cost Optimization:")
    print("-" * 30)
    result = main(['cost', '--budget', '1000', '--optimize', '--experiments', 'cost_experiments.json'])
    print(f"Cost optimization result: {result}")
    
    # Test job scheduling
    print("\n📅 Testing Job Scheduling:")
    print("-" * 30)
    result = main(['schedule', '--goal', 'minimize_cost', '--jobs', 'test_jobs.json'])
    print(f"Schedule result: {result}")
    
    # Test help
    print("\n❓ Testing Help Command:")
    print("-" * 30)
    result = main(['--help'])
    print(f"Help command result: {result}")
    
    # Cleanup
    for filename in ['deploy_config.json', 'test_jobs.json', 'cost_experiments.json']:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("\n" + "=" * 50)
    print("✅ CLI demonstration completed!")

if __name__ == "__main__":
    demo_cli_commands()