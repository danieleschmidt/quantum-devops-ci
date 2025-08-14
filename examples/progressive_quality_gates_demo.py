#!/usr/bin/env python3
"""
Progressive Quality Gates Demo

Demonstrates the Progressive Quality Gates system for quantum DevOps CI/CD
with adaptive thresholds, predictive failure detection, and intelligent progression.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any
import json

# Import the Progressive Quality Gates system
try:
    from src.quantum_devops_ci.progressive_quality_gates import (
        ProgressiveQualityGates,
        QualityGateConfig,
        GateSeverity,
        QualityMetric,
        QualityGateResult,
        GateStatus,
        run_quality_pipeline,
        create_custom_gate
    )
except ImportError:
    # If running as a script, adjust the path
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.quantum_devops_ci.progressive_quality_gates import (
        ProgressiveQualityGates,
        QualityGateConfig,
        GateSeverity,
        QualityMetric,
        QualityGateResult,
        GateStatus,
        run_quality_pipeline,
        create_custom_gate
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def custom_quantum_optimization_gate(context: Dict[str, Any]) -> QualityGateResult:
    """
    Custom quality gate for quantum circuit optimization validation.
    
    This demonstrates how to create custom gates that integrate with
    the Progressive Quality Gates system.
    """
    logger.info("Executing custom quantum optimization gate")
    
    # Simulate optimization analysis
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # Generate realistic optimization metrics
    original_depth = context.get("circuit_depth", 50)
    optimized_depth = int(original_depth * 0.65)  # 35% reduction
    
    original_gates = context.get("circuit_gates", 200)
    optimized_gates = int(original_gates * 0.72)  # 28% reduction
    
    fidelity_improvement = 0.15  # 15% improvement
    cost_reduction = 0.32  # 32% cost reduction
    
    metrics = [
        QualityMetric(
            name="depth_reduction",
            value=(original_depth - optimized_depth) / original_depth,
            threshold=0.2,  # At least 20% reduction
            operator="gt",
            weight=2.0,
            description="Circuit depth reduction percentage"
        ),
        QualityMetric(
            name="gate_reduction", 
            value=(original_gates - optimized_gates) / original_gates,
            threshold=0.15,  # At least 15% reduction
            operator="gt",
            weight=1.5,
            description="Gate count reduction percentage"
        ),
        QualityMetric(
            name="fidelity_improvement",
            value=fidelity_improvement,
            threshold=0.1,  # At least 10% improvement
            operator="gt",
            weight=2.5,
            description="Expected fidelity improvement"
        ),
        QualityMetric(
            name="cost_reduction",
            value=cost_reduction,
            threshold=0.2,  # At least 20% cost reduction
            operator="gt",
            weight=2.0,
            description="Execution cost reduction"
        )
    ]
    
    # Calculate overall score
    passed_metrics = sum(1 for m in metrics if m.evaluate())
    total_weight = sum(m.weight for m in metrics)
    score = sum(m.weight for m in metrics if m.evaluate()) / total_weight
    
    # Determine status
    if score >= 0.85:
        status = GateStatus.PASSED
    elif score >= 0.65:
        status = GateStatus.WARNING
    else:
        status = GateStatus.FAILED
    
    # Generate recommendations based on results
    recommendations = []
    if not metrics[0].evaluate():
        recommendations.append("Consider using transpiler optimization passes to reduce circuit depth")
    if not metrics[1].evaluate():
        recommendations.append("Apply gate fusion techniques to reduce total gate count")
    if not metrics[2].evaluate():
        recommendations.append("Review noise model and consider error mitigation strategies")
    if not metrics[3].evaluate():
        recommendations.append("Optimize backend selection and shot allocation for cost efficiency")
    
    return QualityGateResult(
        gate_id="quantum_optimization_validation",
        status=status,
        score=score,
        metrics=metrics,
        execution_time=0.0,  # Will be set by caller
        timestamp=asyncio.get_event_loop().time(),
        details={
            "original_depth": original_depth,
            "optimized_depth": optimized_depth,
            "original_gates": original_gates,
            "optimized_gates": optimized_gates,
            "optimization_technique": "hybrid_ml_optimization"
        },
        recommendations=recommendations
    )


async def demo_basic_pipeline():
    """Demonstrate basic Progressive Quality Gates pipeline execution."""
    print("\n🚀 === PROGRESSIVE QUALITY GATES DEMO ===")
    print("Demonstrating basic pipeline execution with default gates\n")
    
    # Create context with sample quantum circuit data
    context = {
        "circuit_gate_count": 150,
        "circuit_depth": 32,
        "simulation_fidelity": 0.94,
        "sim_time": 2.3,
        "noise_fidelity": 0.82,
        "error_rate": 0.035,
        "code_coverage": 0.89,
        "throughput": 165.0,
        "latency": 38.0,
        "completed_gates": {}
    }
    
    # Execute the pipeline
    start_time = time.time()
    results = await run_quality_pipeline(context)
    execution_time = time.time() - start_time
    
    print(f"Pipeline executed in {execution_time:.2f} seconds\n")
    
    # Display results
    print("📊 PIPELINE RESULTS:")
    print("=" * 60)
    
    for gate_id, result in results.items():
        status_emoji = {
            GateStatus.PASSED: "✅",
            GateStatus.WARNING: "⚠️",
            GateStatus.FAILED: "❌",
            GateStatus.SKIPPED: "⏭️"
        }
        
        print(f"{status_emoji.get(result.status, '❓')} {gate_id}")
        print(f"   Status: {result.status.value.upper()}")
        print(f"   Score: {result.score:.2%}")
        print(f"   Execution Time: {result.execution_time:.2f}s")
        
        if result.metrics:
            print(f"   Metrics: {len(result.metrics)} evaluated")
            for metric in result.metrics[:2]:  # Show first 2 metrics
                status = "✓" if metric.evaluate() else "✗"
                print(f"     {status} {metric.name}: {metric.value:.3f} (threshold: {metric.threshold})")
        
        print()
    
    # Calculate summary statistics
    total_gates = len(results)
    passed = sum(1 for r in results.values() if r.status == GateStatus.PASSED)
    failed = sum(1 for r in results.values() if r.status == GateStatus.FAILED)
    warnings = sum(1 for r in results.values() if r.status == GateStatus.WARNING)
    skipped = sum(1 for r in results.values() if r.status == GateStatus.SKIPPED)
    
    print("📈 PIPELINE SUMMARY:")
    print("=" * 60)
    print(f"Total Gates: {total_gates}")
    print(f"✅ Passed: {passed} ({passed/total_gates:.1%})")
    print(f"⚠️  Warnings: {warnings} ({warnings/total_gates:.1%})")
    print(f"❌ Failed: {failed} ({failed/total_gates:.1%})")
    print(f"⏭️  Skipped: {skipped} ({skipped/total_gates:.1%})")
    print(f"Overall Success Rate: {(passed + warnings)/total_gates:.1%}")


async def demo_custom_gates():
    """Demonstrate custom gate creation and integration."""
    print("\n🛠️ === CUSTOM GATES DEMO ===")
    print("Demonstrating custom gate creation and pipeline integration\n")
    
    # Create Progressive Quality Gates instance
    gates = ProgressiveQualityGates()
    
    # Register custom gate
    custom_config = create_custom_gate(
        gate_id="quantum_optimization_validation",
        name="Quantum Circuit Optimization Validation",
        handler=custom_quantum_optimization_gate,
        severity=GateSeverity.HIGH,
        dependencies=["performance_benchmarks"]
    )
    
    gates.register_gate(custom_config)
    gates.register_handler("quantum_optimization_validation", custom_quantum_optimization_gate)
    
    print("✅ Registered custom quantum optimization gate")
    
    # Create context with optimization data
    context = {
        "circuit_gate_count": 180,
        "circuit_depth": 45,
        "circuit_gates": 200,
        "simulation_fidelity": 0.91,
        "sim_time": 3.1,
        "noise_fidelity": 0.79,
        "error_rate": 0.042,
        "code_coverage": 0.92,
        "throughput": 140.0,
        "latency": 52.0,
        "completed_gates": {}
    }
    
    # Execute pipeline with custom gate
    print("\n🔄 Executing pipeline with custom gate...")
    results = await gates.execute_pipeline(context)
    
    # Show custom gate results
    if "quantum_optimization_validation" in results:
        result = results["quantum_optimization_validation"]
        print(f"\n🎯 CUSTOM GATE RESULT:")
        print("=" * 40)
        print(f"Status: {result.status.value.upper()}")
        print(f"Score: {result.score:.2%}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.details:
            print("\nOptimization Details:")
            details = result.details
            print(f"  Original Depth: {details.get('original_depth')}")
            print(f"  Optimized Depth: {details.get('optimized_depth')}")
            print(f"  Depth Reduction: {((details.get('original_depth', 0) - details.get('optimized_depth', 0)) / details.get('original_depth', 1)):.1%}")
            print(f"  Gate Reduction: {((details.get('original_gates', 0) - details.get('optimized_gates', 0)) / details.get('original_gates', 1)):.1%}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")


async def demo_adaptive_thresholds():
    """Demonstrate adaptive threshold functionality."""
    print("\n🎯 === ADAPTIVE THRESHOLDS DEMO ===")
    print("Demonstrating adaptive threshold adjustment based on historical performance\n")
    
    gates = ProgressiveQualityGates()
    
    # Simulate multiple pipeline runs to build history
    print("🔄 Simulating historical executions to demonstrate adaptive behavior...")
    
    contexts = []
    for i in range(15):
        # Simulate gradually improving performance
        improvement_factor = 1 + (i * 0.02)  # 2% improvement per iteration
        
        context = {
            "circuit_gate_count": int(120 / improvement_factor),
            "circuit_depth": int(25 / improvement_factor),
            "simulation_fidelity": min(0.98, 0.85 + i * 0.008),
            "sim_time": max(0.5, 3.0 - i * 0.15),
            "noise_fidelity": min(0.92, 0.75 + i * 0.01),
            "error_rate": max(0.005, 0.08 - i * 0.004),
            "code_coverage": min(0.95, 0.82 + i * 0.008),
            "throughput": 100 + i * 8,
            "latency": max(15.0, 80.0 - i * 4),
            "completed_gates": {}
        }
        contexts.append(context)
    
    # Execute historical runs
    for i, context in enumerate(contexts):
        print(f"  Run {i+1}/15: Fidelity={context['simulation_fidelity']:.3f}, "
              f"Coverage={context['code_coverage']:.2%}, "
              f"Error Rate={context['error_rate']:.3f}")
        
        await gates.execute_pipeline(context)
        
        # Small delay to simulate real execution
        await asyncio.sleep(0.1)
    
    print("\n📊 Historical execution completed")
    
    # Show pipeline health metrics
    health = gates.get_pipeline_health()
    
    print(f"\n🩺 PIPELINE HEALTH ANALYSIS:")
    print("=" * 50)
    print(f"Health Status: {health['status'].upper()}")
    print(f"Health Score: {health['health_score']:.2%}")
    print(f"Success Rate: {health['success_rate']:.2%}")
    print(f"Average Score: {health['average_score']:.2%}")
    print(f"Average Execution Time: {health['average_execution_time']:.2f}s")
    print(f"Total Executions: {health['total_executions']}")
    
    # Show adaptive threshold adjustments
    print(f"\n🎛️ ADAPTIVE THRESHOLD ADJUSTMENTS:")
    print("=" * 50)
    for gate_id, adjustments in gates.adaptive_thresholds.items():
        print(f"Gate: {gate_id}")
        print(f"  Adjustment Factor: {adjustments.get('adjustment_factor', 1.0):.3f}")
        print(f"  Success Rate: {adjustments.get('success_rate', 0.0):.2%}")


async def demo_failure_prediction():
    """Demonstrate predictive failure detection capabilities."""
    print("\n🔮 === PREDICTIVE FAILURE ANALYSIS DEMO ===")
    print("Demonstrating ML-based failure prediction and intelligent pipeline decisions\n")
    
    gates = ProgressiveQualityGates()
    
    # Create a context that's likely to cause failures
    problematic_context = {
        "circuit_gate_count": 500,  # Very high gate count
        "circuit_depth": 150,       # Very deep circuit
        "simulation_fidelity": 0.65,  # Low fidelity
        "sim_time": 15.0,           # Slow simulation
        "noise_fidelity": 0.45,     # Poor noise performance
        "error_rate": 0.15,         # High error rate
        "code_coverage": 0.60,      # Low coverage
        "throughput": 25.0,         # Poor throughput
        "latency": 250.0,           # High latency
        "completed_gates": {}
    }
    
    print("🚨 Executing pipeline with problematic parameters...")
    print(f"   Circuit Depth: {problematic_context['circuit_depth']} (high)")
    print(f"   Gate Count: {problematic_context['circuit_gate_count']} (very high)")
    print(f"   Error Rate: {problematic_context['error_rate']:.1%} (high)")
    print(f"   Simulation Fidelity: {problematic_context['simulation_fidelity']:.2%} (low)")
    
    results = await gates.execute_pipeline(problematic_context)
    
    # Analyze failure patterns
    failed_gates = [gate_id for gate_id, result in results.items() if result.status == GateStatus.FAILED]
    warning_gates = [gate_id for gate_id, result in results.items() if result.status == GateStatus.WARNING]
    
    print(f"\n⚠️ FAILURE ANALYSIS:")
    print("=" * 40)
    print(f"Failed Gates: {len(failed_gates)}")
    for gate_id in failed_gates:
        print(f"  ❌ {gate_id}")
    
    print(f"\nWarning Gates: {len(warning_gates)}")
    for gate_id in warning_gates:
        print(f"  ⚠️ {gate_id}")
    
    # Show pipeline decision making
    total_gates = len(results)
    failure_rate = len(failed_gates) / total_gates if total_gates > 0 else 0
    
    print(f"\n🤖 INTELLIGENT PIPELINE DECISIONS:")
    print("=" * 50)
    print(f"Overall Failure Rate: {failure_rate:.1%}")
    
    if failure_rate > 0.3:
        print("🛑 Decision: Pipeline termination would be recommended")
        print("   Reason: High failure rate indicates systemic issues")
        print("   Recommendation: Review circuit design and parameters")
    elif failure_rate > 0.1:
        print("⚠️ Decision: Continue with warnings")
        print("   Reason: Moderate failures, some gates still passing")
        print("   Recommendation: Monitor closely and consider optimizations")
    else:
        print("✅ Decision: Continue pipeline execution")
        print("   Reason: Low failure rate, pipeline is healthy")


async def main():
    """Main demo function."""
    print("🌌 PROGRESSIVE QUALITY GATES FOR QUANTUM DEVOPS CI/CD")
    print("=" * 60)
    print("Demonstrating advanced quality gate system with:")
    print("• Adaptive threshold adjustment")
    print("• Predictive failure detection") 
    print("• Intelligent gate progression")
    print("• Custom gate integration")
    print("• Pipeline health monitoring")
    
    try:
        # Run all demos
        await demo_basic_pipeline()
        await demo_custom_gates()
        await demo_adaptive_thresholds()
        await demo_failure_prediction()
        
        print("\n🎉 === DEMO COMPLETED SUCCESSFULLY ===")
        print("Progressive Quality Gates system demonstration complete!")
        print("Ready for production deployment in quantum DevOps pipelines.")
        
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