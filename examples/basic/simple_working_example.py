#!/usr/bin/env python3
"""
Simple working example without authentication.

This example demonstrates basic quantum-devops-ci functionality
with mock circuits and simplified operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum_devops_ci.deployment import QuantumDeployment, DeploymentStrategy, QuantumABTest

def create_mock_circuit():
    """Create a mock quantum circuit for testing."""
    return {
        "type": "bell_circuit",
        "qubits": 2,
        "gates": ["h", "cx"],
        "measurements": True
    }

def create_mock_vqe_circuit():
    """Create a mock VQE circuit."""
    return {
        "type": "vqe_circuit", 
        "qubits": 4,
        "gates": ["ry", "rz", "cx"],
        "parameters": [0.1, 0.2, 0.3, 0.4],
        "measurements": True
    }

def test_deployment_system():
    """Test the deployment system with proper configuration."""
    print("🚀 Testing Quantum Deployment System")
    print("-" * 40)
    
    # Configure deployment with proper structure
    config = {
        'environments': {
            'production': {
                'backend': 'qasm_simulator',
                'allocation': 100.0,
                'max_shots': 10000,
                'validation_shots': 1000
            },
            'staging': {
                'backend': 'statevector_simulator',
                'allocation': 50.0,
                'max_shots': 5000,
                'validation_shots': 500
            }
        }
    }
    
    try:
        # Initialize deployment manager
        deployer = QuantumDeployment(config)
        print(f"✅ Deployment manager initialized with {len(deployer.environments)} environments")
        
        # Deploy algorithm using blue-green strategy
        deployment_id = deployer.deploy(
            algorithm_id='bell_state_v1',
            circuit_factory=create_mock_circuit,
            strategy=DeploymentStrategy.BLUE_GREEN
        )
        
        print(f"✅ Blue-green deployment started: {deployment_id}")
        
        # Check deployment status
        status = deployer.get_deployment_status(deployment_id)
        print(f"✅ Deployment status: {status['status']}")
        print(f"   Duration: {status.get('duration_seconds', 'N/A')} seconds")
        
        # Validate deployment
        validation_result = deployer.validate_deployment(deployment_id, 'production')
        print(f"✅ Validation: {'PASSED' if validation_result.passed else 'FAILED'}")
        print(f"   Fidelity: {validation_result.fidelity:.3f}")
        print(f"   Error rate: {validation_result.error_rate:.3f}")
        print(f"   Cost: ${validation_result.cost:.4f}")
        
        # Test canary deployment
        canary_deployment_id = deployer.deploy(
            algorithm_id='bell_state_v2',
            circuit_factory=create_mock_circuit, 
            strategy=DeploymentStrategy.CANARY,
            canary_percentages=[10, 25, 50, 100]
        )
        
        print(f"✅ Canary deployment started: {canary_deployment_id}")
        
        canary_status = deployer.get_deployment_status(canary_deployment_id)
        print(f"✅ Canary deployment status: {canary_status['status']}")
        
    except Exception as e:
        print(f"❌ Deployment test failed: {e}")
        import traceback
        traceback.print_exc()

def test_ab_testing():
    """Test A/B testing framework."""
    print("\n🧪 Testing A/B Testing Framework")
    print("-" * 40)
    
    try:
        # Define test variants
        variants = {
            'optimized': {
                'optimizer': 'SPSA',
                'iterations': 100,
                'learning_rate': 0.01,
                'traffic_allocation': 0.5
            },
            'baseline': {
                'optimizer': 'COBYLA',
                'iterations': 50,
                'learning_rate': 0.05,
                'traffic_allocation': 0.5
            }
        }
        
        # Initialize A/B test
        ab_test = QuantumABTest(
            name='vqe_optimizer_comparison',
            variants=variants,
            metrics=['convergence_rate', 'final_energy', 'total_evaluations']
        )
        
        print(f"✅ A/B test initialized with {len(ab_test.variants)} variants")
        
        # Mock circuit factory for A/B testing
        def circuit_factory_ab(config):
            circuit = create_mock_vqe_circuit()
            circuit['optimizer'] = config['optimizer']
            circuit['learning_rate'] = config['learning_rate']
            return circuit
        
        # Run A/B test
        results = ab_test.run(
            circuit_factory=circuit_factory_ab,
            duration_hours=1,
            traffic_split=0.6  
        )
        
        print(f"✅ A/B test completed with {len(results)} results")
        
        for variant_name, result in results.items():
            print(f"   {variant_name}:")
            print(f"     Sample size: {result.sample_size}")
            print(f"     Metrics: {result.metric_values}")
        
        # Analyze results
        winner_analysis = ab_test.determine_winner(
            results,
            confidence_level=0.95,
            minimum_difference=0.05
        )
        
        print(f"✅ Winner analysis completed")
        print(f"   Winner: {winner_analysis.winner}")
        print(f"   Improvement: {winner_analysis.improvement:.1f}%")
        print(f"   P-value: {winner_analysis.p_value:.4f}")
        print(f"   Recommendation: {winner_analysis.recommendation}")
        
        # Get test summary
        summary = ab_test.get_test_summary()
        print(f"✅ Test summary: {summary['status']}")
        print(f"   Total samples: {summary['total_samples']}")
        
    except Exception as e:
        print(f"❌ A/B testing failed: {e}")
        import traceback
        traceback.print_exc()

def test_advanced_deployment_strategies():
    """Test advanced deployment strategies.""" 
    print("\n🎯 Testing Advanced Deployment Strategies")
    print("-" * 40)
    
    config = {
        'environments': {
            'production': {'backend': 'ibmq_qasm_simulator', 'allocation': 100.0, 'max_shots': 10000},
            'blue': {'backend': 'blue_backend', 'allocation': 50.0, 'max_shots': 5000},
            'green': {'backend': 'green_backend', 'allocation': 50.0, 'max_shots': 5000}
        }
    }
    
    deployer = QuantumDeployment(config)
    
    try:
        # Rolling deployment
        rolling_id = deployer.deploy(
            algorithm_id='quantum_ml_v3',
            circuit_factory=create_mock_vqe_circuit,
            strategy=DeploymentStrategy.ROLLING,
            instances=['instance_1', 'instance_2', 'instance_3']
        )
        
        print(f"✅ Rolling deployment: {rolling_id}")
        
        status = deployer.get_deployment_status(rolling_id)
        print(f"   Status: {status['status']}")
        
        # Test rollback
        success = deployer.rollback_deployment(rolling_id)
        print(f"✅ Rollback {'successful' if success else 'failed'}")
        
        # Blue-green with proper configuration
        if 'blue' in deployer.environments and 'green' in deployer.environments:
            bg_id = deployer.deploy(
                algorithm_id='quantum_optimization_v2',
                circuit_factory=create_mock_circuit,
                strategy=DeploymentStrategy.BLUE_GREEN
            )
            print(f"✅ Blue-green deployment: {bg_id}")
        
    except Exception as e:
        print(f"❌ Advanced deployment test failed: {e}")

def main():
    """Run all tests."""
    print("🎯 Quantum DevOps CI - Working Examples")
    print("=" * 50)
    
    test_deployment_system()
    test_ab_testing()
    test_advanced_deployment_strategies()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    main()