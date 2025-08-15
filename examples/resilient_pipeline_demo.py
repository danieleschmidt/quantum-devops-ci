#!/usr/bin/env python3
"""
Resilient Pipeline Demo

Demonstrates advanced error handling, circuit breakers, retry strategies,
and failure recovery mechanisms for quantum DevOps CI/CD pipelines.
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any

# Import the Resilient Pipeline system
try:
    from src.quantum_devops_ci.resilient_pipeline import (
        ResilientPipeline,
        RetryConfig,
        RetryStrategy,
        ErrorSeverity,
        ValidationRule,
        CircuitBreakerConfig,
        create_resilient_pipeline,
        ValidationError
    )
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.quantum_devops_ci.resilient_pipeline import (
        ResilientPipeline,
        RetryConfig,
        RetryStrategy,
        ErrorSeverity,
        ValidationRule,
        CircuitBreakerConfig,
        create_resilient_pipeline,
        ValidationError
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulatedFailure(Exception):
    """Simulated failure for demonstration."""
    pass


async def unreliable_quantum_simulation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulated quantum simulation that sometimes fails.
    
    This demonstrates how the resilient pipeline handles:
    - Intermittent failures
    - Timeout errors
    - Connection issues
    """
    attempt = context.get("attempt", 1)
    correlation_id = context.get("correlation_id", "unknown")
    
    logger.info(f"Running quantum simulation (attempt {attempt}, ID: {correlation_id})")
    
    # Simulate processing time
    await asyncio.sleep(0.2)
    
    # Simulate various failure scenarios
    failure_probability = 0.4 - (attempt * 0.1)  # Lower chance on retry
    
    if random.random() < failure_probability:
        failure_type = random.choice([
            "connection_error",
            "timeout_error", 
            "memory_error",
            "quantum_error"
        ])
        
        if failure_type == "connection_error":
            raise ConnectionError("Failed to connect to quantum backend")
        elif failure_type == "timeout_error":
            raise TimeoutError("Quantum simulation timed out")
        elif failure_type == "memory_error":
            raise MemoryError("Insufficient memory for simulation")
        else:
            raise SimulatedFailure("Quantum coherence lost during simulation")
    
    # Success case
    circuit_depth = context.get("circuit_depth", 20)
    shots = context.get("shots", 1000)
    
    # Simulate realistic results
    fidelity = 0.95 - (circuit_depth * 0.001)  # Fidelity decreases with depth
    execution_time = shots * 0.001  # 1ms per shot
    
    return {
        "simulation_successful": True,
        "fidelity": fidelity,
        "execution_time": execution_time,
        "shots_completed": shots,
        "backend": "aer_simulator",
        "noise_model": "ibmq_manhattan",
        "correlation_id": correlation_id
    }


async def flaky_cost_optimizer(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulated cost optimizer that occasionally fails.
    
    Demonstrates circuit breaker behavior when a service
    becomes consistently unreliable.
    """
    attempt = context.get("attempt", 1)
    correlation_id = context.get("correlation_id", "unknown")
    
    logger.info(f"Running cost optimization (attempt {attempt}, ID: {correlation_id})")
    
    # Simulate heavy processing
    await asyncio.sleep(0.3)
    
    # High failure rate to trigger circuit breaker
    if random.random() < 0.7:
        raise ConnectionError("Cost optimization service unavailable")
    
    # Success case
    gate_count = context.get("gate_count", 100)
    providers = ["ibmq", "aws_braket", "ionq", "rigetti"]
    
    # Calculate costs for different providers
    costs = {}
    for provider in providers:
        base_cost = gate_count * random.uniform(0.01, 0.05)
        costs[provider] = round(base_cost, 2)
    
    # Find optimal provider
    optimal_provider = min(costs.items(), key=lambda x: x[1])
    
    return {
        "optimization_successful": True,
        "provider_costs": costs,
        "recommended_provider": optimal_provider[0],
        "estimated_savings": round(max(costs.values()) - optimal_provider[1], 2),
        "correlation_id": correlation_id
    }


async def circuit_compiler_with_validation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Circuit compiler that includes input validation.
    
    Demonstrates validation rule integration and security checking.
    """
    correlation_id = context.get("correlation_id", "unknown")
    
    logger.info(f"Compiling quantum circuit (ID: {correlation_id})")
    
    # Simulate compilation time
    await asyncio.sleep(0.1)
    
    # Check for invalid circuit parameters
    circuit_depth = context.get("circuit_depth", 0)
    if circuit_depth > 150:
        raise ValidationError(f"Circuit depth {circuit_depth} exceeds maximum allowed (150)")
    
    gate_count = context.get("gate_count", 0)
    if gate_count > 800:
        raise ValidationError(f"Gate count {gate_count} exceeds maximum allowed (800)")
    
    # Simulate successful compilation
    return {
        "compilation_successful": True,
        "compiled_circuit_depth": circuit_depth,
        "compiled_gate_count": gate_count,
        "optimization_level": context.get("optimization_level", 1),
        "target_backend": context.get("target_backend", "qasm_simulator"),
        "correlation_id": correlation_id
    }


async def demo_basic_resilience():
    """Demonstrate basic resilience features."""
    print("\n🛡️ === BASIC RESILIENCE DEMO ===")
    print("Demonstrating retry strategies and error recovery\n")
    
    pipeline = await create_resilient_pipeline("demo_pipeline")
    
    # Test context with reasonable parameters
    context = {
        "circuit_depth": 25,
        "gate_count": 120,
        "shots": 1000,
        "qubit_count": 5,
        "optimization_level": 2
    }
    
    print("🔄 Testing unreliable quantum simulation with retry...")
    
    try:
        result = await pipeline.execute_with_resilience(
            operation=unreliable_quantum_simulation,
            component="quantum_simulation",
            context=context
        )
        
        print("✅ Simulation completed successfully!")
        print(f"   Fidelity: {result['fidelity']:.3f}")
        print(f"   Execution Time: {result['execution_time']:.3f}s")
        print(f"   Shots: {result['shots_completed']}")
        
    except Exception as e:
        print(f"❌ Simulation failed after all retries: {e}")
    
    # Show pipeline health
    health = pipeline.get_pipeline_health()
    print(f"\n📊 Pipeline Health: {health['overall_status'].upper()}")
    print(f"   Success Rate: {health['success_rate']:.2%}")
    print(f"   Total Executions: {health['total_executions']}")


async def demo_circuit_breaker():
    """Demonstrate circuit breaker functionality."""
    print("\n⚡ === CIRCUIT BREAKER DEMO ===")
    print("Demonstrating circuit breaker pattern with flaky service\n")
    
    pipeline = await create_resilient_pipeline("breaker_demo")
    
    # Configure aggressive circuit breaker for demonstration
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=3,    # Open after 3 failures
        recovery_timeout=2.0,   # Try recovery after 2 seconds
        success_threshold=2     # Close after 2 successes
    )
    
    # Override the circuit breaker for cost optimization
    from src.quantum_devops_ci.resilient_pipeline import CircuitBreaker
    pipeline.circuit_breakers["cost_optimization"] = CircuitBreaker(
        "cost_optimization", 
        circuit_breaker_config
    )
    
    context = {
        "gate_count": 200,
        "circuit_depth": 30
    }
    
    print("🔄 Testing flaky cost optimizer to trigger circuit breaker...")
    
    # Try multiple executions to trigger circuit breaker
    for i in range(8):
        print(f"\n--- Execution {i+1} ---")
        
        try:
            result = await pipeline.execute_with_resilience(
                operation=flaky_cost_optimizer,
                component="cost_optimization", 
                context=context,
                custom_retry_config=RetryConfig(max_attempts=1)  # No retries for this demo
            )
            
            print("✅ Cost optimization succeeded!")
            print(f"   Recommended: {result['recommended_provider']}")
            print(f"   Savings: ${result['estimated_savings']:.2f}")
            
        except Exception as e:
            print(f"❌ Cost optimization failed: {e}")
        
        # Show circuit breaker status
        cb_status = pipeline.circuit_breakers["cost_optimization"].get_status()
        print(f"   Circuit Breaker: {cb_status['state']} (failures: {cb_status['failure_count']})")
        
        # Small delay between attempts
        await asyncio.sleep(0.5)
    
    print(f"\n📈 Final Circuit Breaker Status:")
    cb_status = pipeline.circuit_breakers["cost_optimization"].get_status()
    print(f"   State: {cb_status['state']}")
    print(f"   Failure Count: {cb_status['failure_count']}")
    print(f"   Recent Success Rate: {cb_status['recent_success_rate']:.2%}")


async def demo_validation_rules():
    """Demonstrate input validation and security checking."""
    print("\n🔒 === VALIDATION RULES DEMO ===")
    print("Demonstrating input validation and security checks\n")
    
    pipeline = await create_resilient_pipeline("validation_demo")
    
    # Test case 1: Valid parameters
    print("🧪 Test 1: Valid circuit parameters")
    valid_context = {
        "circuit_depth": 50,
        "gate_count": 200,
        "qubit_count": 8,
        "operations": ["measurement", "rotation"]
    }
    
    validation_errors = await pipeline.validate_input(valid_context, "circuit_compilation")
    
    if not validation_errors:
        print("✅ Validation passed!")
        
        try:
            result = await pipeline.execute_with_resilience(
                operation=circuit_compiler_with_validation,
                component="circuit_compilation",
                context=valid_context
            )
            print(f"   Compilation successful for {result['compiled_gate_count']} gates")
        except Exception as e:
            print(f"❌ Compilation failed: {e}")
    else:
        print("❌ Validation failed:")
        for error in validation_errors:
            print(f"   {error.severity.value}: {error.error_message}")
    
    # Test case 2: Invalid parameters (too complex)
    print("\n🧪 Test 2: Invalid circuit parameters (too complex)")
    invalid_context = {
        "circuit_depth": 250,  # Exceeds limit
        "gate_count": 1200,    # Exceeds limit
        "qubit_count": 60,     # Exceeds limit
        "operations": ["measurement"]
    }
    
    validation_errors = await pipeline.validate_input(invalid_context, "circuit_compilation")
    
    if validation_errors:
        print("❌ Validation failed (as expected):")
        for error in validation_errors:
            print(f"   {error.severity.value}: {error.error_message}")
    
    # Try to compile anyway to show validation in action
    try:
        result = await pipeline.execute_with_resilience(
            operation=circuit_compiler_with_validation,
            component="circuit_compilation",
            context=invalid_context
        )
    except Exception as e:
        print(f"   Compilation blocked: {e}")
    
    # Test case 3: Security validation
    print("\n🧪 Test 3: Security validation (potential secrets)")
    security_context = {
        "circuit_depth": 30,
        "gate_count": 150,
        "api_key": "super_secret_key_12345",  # This should trigger security warning
        "password": "hidden_password",         # This too
        "operations": ["system_call"]          # Dangerous operation
    }
    
    validation_errors = await pipeline.validate_input(security_context, "circuit_compilation")
    
    if validation_errors:
        print("🚨 Security issues detected:")
        for error in validation_errors:
            print(f"   {error.severity.value}: {error.error_message}")


async def demo_failure_recovery():
    """Demonstrate failure recovery mechanisms."""
    print("\n🚑 === FAILURE RECOVERY DEMO ===")
    print("Demonstrating fallback strategies and degraded mode execution\n")
    
    pipeline = await create_resilient_pipeline("recovery_demo")
    
    # Simulate a scenario where primary quantum simulation fails
    # but fallback mechanisms can provide alternative results
    
    async def always_failing_simulation(context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulation that always fails to demonstrate recovery."""
        raise ConnectionError("Primary quantum backend unavailable")
    
    context = {
        "circuit_depth": 40,
        "gate_count": 180,
        "shots": 2000
    }
    
    print("🔄 Testing simulation with forced failure (recovery expected)...")
    
    try:
        result = await pipeline.execute_with_resilience(
            operation=always_failing_simulation,
            component="quantum_simulation",
            context=context
        )
        
        print("✅ Recovery successful!")
        print(f"   Mode: {result.get('mode', 'normal')}")
        print(f"   Fidelity: {result.get('fidelity', 'N/A')}")
        
        if result.get('warning'):
            print(f"   ⚠️ Warning: {result['warning']}")
            
    except Exception as e:
        print(f"❌ All recovery attempts failed: {e}")
    
    # Test cost calculation fallback
    print("\n🔄 Testing cost calculation with fallback...")
    
    async def failing_cost_calc(context: Dict[str, Any]) -> Dict[str, Any]:
        """Cost calculation that always fails."""
        raise TimeoutError("Advanced cost optimization service timeout")
    
    try:
        result = await pipeline.execute_with_resilience(
            operation=failing_cost_calc,
            component="cost_calculation",
            context=context
        )
        
        print("✅ Cost calculation fallback successful!")
        print(f"   Estimated Cost: ${result.get('estimated_cost', 'N/A'):.2f}")
        print(f"   Mode: {result.get('mode', 'normal')}")
        print(f"   Accuracy: {result.get('accuracy', 'high')}")
        
        if result.get('warning'):
            print(f"   ⚠️ Warning: {result['warning']}")
            
    except Exception as e:
        print(f"❌ Cost calculation failed: {e}")


async def demo_comprehensive_monitoring():
    """Demonstrate comprehensive pipeline monitoring and analysis."""
    print("\n📊 === COMPREHENSIVE MONITORING DEMO ===")
    print("Demonstrating pipeline health monitoring and error analysis\n")
    
    pipeline = await create_resilient_pipeline("monitoring_demo")
    
    # Run multiple operations to generate metrics
    operations = [
        (unreliable_quantum_simulation, "quantum_simulation"),
        (circuit_compiler_with_validation, "circuit_compilation"),
        (flaky_cost_optimizer, "cost_optimization")
    ]
    
    print("🔄 Running multiple operations to generate monitoring data...")
    
    for i in range(12):
        for operation, component in operations:
            context = {
                "circuit_depth": random.randint(20, 80),
                "gate_count": random.randint(100, 400),
                "shots": random.randint(500, 2000),
                "correlation_id": f"{component}_{i}_{int(time.time())}"
            }
            
            try:
                await pipeline.execute_with_resilience(
                    operation=operation,
                    component=component,
                    context=context,
                    custom_retry_config=RetryConfig(max_attempts=2)
                )
            except Exception:
                pass  # Ignore failures for monitoring demo
        
        # Small delay between batches
        await asyncio.sleep(0.1)
    
    print("✅ Monitoring data generated")
    
    # Show comprehensive health report
    health = pipeline.get_pipeline_health()
    
    print(f"\n🩺 PIPELINE HEALTH REPORT:")
    print("=" * 50)
    print(f"Overall Status: {health['overall_status'].upper()}")
    print(f"Success Rate: {health['success_rate']:.2%}")
    print(f"Total Executions: {health['total_executions']}")
    print(f"Total Failures: {health['total_failures']}")
    print(f"Recent Errors (1h): {health['recent_errors_count']}")
    print(f"Critical Errors: {health['critical_errors_count']}")
    
    print(f"\n🏗️ COMPONENT HEALTH:")
    for component, health_data in health['component_health'].items():
        status = health_data.get('status', 'unknown')
        print(f"   {component}: {status.upper()}")
        
        if 'last_error' in health_data:
            print(f"      Last Error: {health_data['last_error'][:50]}...")
    
    print(f"\n⚡ CIRCUIT BREAKER STATUS:")
    for name, cb_status in health['circuit_breaker_status'].items():
        print(f"   {name}: {cb_status['state']} (failures: {cb_status['failure_count']})")
    
    # Show error analysis
    error_analysis = pipeline.get_error_analysis()
    
    if error_analysis.get('total_errors', 0) > 0:
        print(f"\n🔍 ERROR ANALYSIS:")
        print("=" * 30)
        print(f"Total Errors: {error_analysis['total_errors']}")
        print(f"Most Problematic Component: {error_analysis['most_problematic_component']}")
        
        print(f"\nError Distribution by Type:")
        for error_type, count in error_analysis['error_distribution']['by_type'].items():
            print(f"   {error_type}: {count}")
        
        print(f"\nTop Recommendations:")
        for rec in error_analysis['top_recommendations']:
            print(f"   • {rec}")


async def main():
    """Main demo function."""
    print("🛡️ RESILIENT PIPELINE FOR QUANTUM DEVOPS CI/CD")
    print("=" * 60)
    print("Demonstrating advanced error handling and recovery:")
    print("• Intelligent retry strategies")
    print("• Circuit breaker fault tolerance")
    print("• Input validation and security") 
    print("• Failure recovery mechanisms")
    print("• Comprehensive monitoring")
    
    try:
        # Run all demos
        await demo_basic_resilience()
        await demo_circuit_breaker()
        await demo_validation_rules()
        await demo_failure_recovery()
        await demo_comprehensive_monitoring()
        
        print("\n🎉 === RESILIENT PIPELINE DEMO COMPLETED ===")
        print("All resilience features demonstrated successfully!")
        print("Pipeline ready for production quantum DevOps workflows.")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)