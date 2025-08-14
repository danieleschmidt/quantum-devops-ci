#!/usr/bin/env python3
"""
Quantum Performance Optimizer Demo

Demonstrates advanced performance optimization, auto-scaling, load balancing,
and intelligent scheduling for quantum DevOps CI/CD workloads.
"""

import asyncio
import logging
import sys
import time
import random
from typing import List

# Import the Quantum Performance Optimizer
try:
    from src.quantum_devops_ci.quantum_performance_optimizer import (
        QuantumPerformanceOptimizer,
        WorkloadSpec,
        OptimizationStrategy,
        ScalingConfig,
        PerformanceMetrics,
        create_quantum_optimizer,
        create_workload_spec
    )
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.quantum_devops_ci.quantum_performance_optimizer import (
        QuantumPerformanceOptimizer,
        WorkloadSpec,
        OptimizationStrategy,
        ScalingConfig,
        PerformanceMetrics,
        create_quantum_optimizer,
        create_workload_spec
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_load_balancing():
    """Demonstrate intelligent load balancing across quantum instances."""
    print("\n⚖️ === LOAD BALANCING DEMO ===")
    print("Demonstrating intelligent workload distribution across quantum instances\n")
    
    # Create optimizer with multiple instances
    optimizer = await create_quantum_optimizer(min_instances=2, max_instances=5)
    
    # Create diverse workloads with different resource requirements
    workloads = [
        create_workload_spec("light_circuit_1", priority=2, cpu_estimate=1.0, memory_estimate=512),
        create_workload_spec("heavy_simulation_1", priority=3, cpu_estimate=4.0, memory_estimate=2048),
        create_workload_spec("medium_optimization_1", priority=2, cpu_estimate=2.0, memory_estimate=1024),
        create_workload_spec("critical_validation_1", priority=5, cpu_estimate=1.5, memory_estimate=768),
        create_workload_spec("batch_processing_1", priority=1, cpu_estimate=3.0, memory_estimate=1536),
    ]
    
    print("🔄 Submitting workloads with different resource requirements...")
    
    # Submit workloads
    submitted_ids = []
    for workload in workloads:
        workload_id = await optimizer.submit_quantum_workload(
            workload, 
            OptimizationStrategy.BALANCED
        )
        submitted_ids.append(workload_id)
        print(f"   ✓ Submitted {workload.workload_id} (CPU: {workload.estimated_cpu}, Priority: {workload.priority})")
    
    # Execute workloads
    print(f"\n🚀 Executing {len(workloads)} workloads with intelligent load balancing...")
    results = await optimizer.execute_workload_batch(workloads, max_concurrent=3)
    
    # Show results
    successful = sum(1 for r in results if r.get("status") == "success")
    total_time = sum(r.get("execution_time", 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"\n📊 LOAD BALANCING RESULTS:")
    print("=" * 40)
    print(f"Successful Workloads: {successful}/{len(workloads)} ({successful/len(workloads):.1%})")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Average Execution Time: {avg_time:.2f}s")
    
    # Show cluster metrics
    status = optimizer.get_comprehensive_status()
    cluster = status["cluster"]
    
    print(f"\n🏗️ CLUSTER UTILIZATION:")
    print(f"   Active Instances: {cluster['healthy_instances']}/{cluster['total_instances']}")
    print(f"   CPU Utilization: {cluster['cluster_utilization']['cpu']:.1f}%")
    print(f"   Memory Utilization: {cluster['cluster_utilization']['memory']:.1f}%")
    
    await optimizer.stop_monitoring()


async def demo_auto_scaling():
    """Demonstrate auto-scaling based on workload demands."""
    print("\n📈 === AUTO-SCALING DEMO ===")
    print("Demonstrating dynamic scaling based on workload demands\n")
    
    # Create optimizer with aggressive scaling settings for demo
    scaling_config = ScalingConfig(
        min_instances=1,
        max_instances=6,
        scale_up_threshold=60.0,    # Lower threshold for demo
        scale_down_threshold=20.0,  # Higher threshold for demo
        scale_up_cooldown=5.0,      # Faster scaling for demo
        scale_down_cooldown=10.0
    )
    
    optimizer = QuantumPerformanceOptimizer(scaling_config)
    await optimizer.start_monitoring(interval=2.0)  # Fast monitoring for demo
    
    print("🔄 Phase 1: Low load (should maintain minimum instances)")
    
    # Submit a few light workloads
    light_workloads = [
        create_workload_spec(f"light_workload_{i}", priority=1, cpu_estimate=0.5, duration_estimate=3.0)
        for i in range(2)
    ]
    
    await optimizer.execute_workload_batch(light_workloads, max_concurrent=2)
    await asyncio.sleep(3)  # Allow monitoring to react
    
    status = optimizer.get_comprehensive_status()
    print(f"   Current Instances: {status['auto_scaling']['current_instances']}")
    print(f"   CPU Utilization: {status['cluster']['cluster_utilization']['cpu']:.1f}%")
    
    print("\n🔄 Phase 2: High load (should trigger scale-up)")
    
    # Submit many heavy workloads to trigger scaling
    heavy_workloads = [
        create_workload_spec(f"heavy_workload_{i}", priority=2, cpu_estimate=3.0, duration_estimate=5.0)
        for i in range(8)
    ]
    
    # Submit workloads without waiting for completion to build queue
    for workload in heavy_workloads:
        await optimizer.submit_quantum_workload(workload)
    
    # Wait for auto-scaler to react
    print("   Waiting for auto-scaler to detect high load...")
    await asyncio.sleep(8)
    
    status = optimizer.get_comprehensive_status()
    scaling_history = status['auto_scaling']['recent_scaling_actions']
    
    print(f"   Current Instances: {status['auto_scaling']['current_instances']}")
    print(f"   Pending Workloads: {status['scheduling']['pending_workloads']}")
    
    if scaling_history:
        latest_action = scaling_history[-1]
        print(f"   Latest Scaling Action: {latest_action['direction']} to {latest_action['instances']} instances")
    
    print("\n🔄 Phase 3: Load completion (should trigger scale-down)")
    
    # Wait for workloads to complete and load to decrease
    print("   Waiting for workloads to complete and scale-down...")
    await asyncio.sleep(15)
    
    final_status = optimizer.get_comprehensive_status()
    final_scaling = final_status['auto_scaling']['recent_scaling_actions']
    
    print(f"   Final Instances: {final_status['auto_scaling']['current_instances']}")
    print(f"   CPU Utilization: {final_status['cluster']['cluster_utilization']['cpu']:.1f}%")
    
    print(f"\n📊 SCALING HISTORY:")
    for action in final_scaling[-3:]:  # Show last 3 actions
        print(f"   {action['timestamp'][:19]}: {action['direction']} to {action['instances']} instances")
    
    await optimizer.stop_monitoring()


async def demo_optimization_strategies():
    """Demonstrate different optimization strategies."""
    print("\n🎯 === OPTIMIZATION STRATEGIES DEMO ===")
    print("Demonstrating different optimization strategies for various workload types\n")
    
    optimizer = await create_quantum_optimizer(min_instances=3, max_instances=5)
    
    # Create workloads for different optimization strategies
    workload_sets = {
        OptimizationStrategy.THROUGHPUT: [
            create_workload_spec(f"throughput_job_{i}", priority=2, cpu_estimate=1.0, duration_estimate=4.0)
            for i in range(4)
        ],
        OptimizationStrategy.LATENCY: [
            create_workload_spec(f"latency_critical_{i}", priority=4, cpu_estimate=1.5, duration_estimate=2.0, deadline_minutes=5)
            for i in range(2)
        ],
        OptimizationStrategy.COST: [
            create_workload_spec(f"cost_optimized_{i}", priority=1, cpu_estimate=2.0, duration_estimate=8.0)
            for i in range(3)
        ],
        OptimizationStrategy.BALANCED: [
            create_workload_spec(f"balanced_workload_{i}", priority=3, cpu_estimate=1.5, duration_estimate=5.0)
            for i in range(3)
        ]
    }
    
    strategy_results = {}
    
    for strategy, workloads in workload_sets.items():
        print(f"\n🔄 Testing {strategy.value.upper()} optimization strategy...")
        
        start_time = time.time()
        
        # Submit all workloads with the specific strategy
        for workload in workloads:
            await optimizer.submit_quantum_workload(workload, strategy)
        
        # Execute workloads
        results = await optimizer.execute_workload_batch(workloads, max_concurrent=3)
        
        execution_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("status") == "success")
        
        # Calculate strategy-specific metrics
        if results and successful > 0:
            avg_latency = sum(
                r.get("metrics", PerformanceMetrics()).latency 
                for r in results if r.get("status") == "success"
            ) / successful
            
            total_cost = sum(
                r.get("metrics", PerformanceMetrics()).cost_per_operation 
                for r in results if r.get("status") == "success"
            )
            
            throughput = successful / execution_time if execution_time > 0 else 0
        else:
            avg_latency = total_cost = throughput = 0
        
        strategy_results[strategy] = {
            "workloads": len(workloads),
            "successful": successful,
            "total_time": execution_time,
            "avg_latency": avg_latency,
            "total_cost": total_cost,
            "throughput": throughput
        }
        
        print(f"   ✓ Completed {successful}/{len(workloads)} workloads in {execution_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} ops/sec")
        print(f"   Avg Latency: {avg_latency:.0f}ms")
        print(f"   Total Cost: ${total_cost:.2f}")
    
    # Compare strategies
    print(f"\n📊 OPTIMIZATION STRATEGY COMPARISON:")
    print("=" * 60)
    print(f"{'Strategy':<12} {'Throughput':<11} {'Latency':<9} {'Cost':<8} {'Success':<7}")
    print("-" * 60)
    
    for strategy, metrics in strategy_results.items():
        print(f"{strategy.value:<12} "
              f"{metrics['throughput']:<11.2f} "
              f"{metrics['avg_latency']:<9.0f} "
              f"${metrics['total_cost']:<7.2f} "
              f"{metrics['successful']}/{metrics['workloads']}")
    
    await optimizer.stop_monitoring()


async def demo_performance_monitoring():
    """Demonstrate comprehensive performance monitoring and analysis."""
    print("\n📊 === PERFORMANCE MONITORING DEMO ===")
    print("Demonstrating performance profiling, bottleneck detection, and optimization recommendations\n")
    
    optimizer = await create_quantum_optimizer(min_instances=2, max_instances=4)
    
    print("🔄 Running diverse workloads to generate performance data...")
    
    # Generate diverse workloads to create interesting performance patterns
    varied_workloads = []
    
    # Some fast, low-resource workloads
    for i in range(5):
        varied_workloads.append(
            create_workload_spec(f"fast_job_{i}", priority=2, cpu_estimate=0.5, duration_estimate=1.0)
        )
    
    # Some slow, high-resource workloads
    for i in range(3):
        varied_workloads.append(
            create_workload_spec(f"slow_job_{i}", priority=3, cpu_estimate=3.0, duration_estimate=8.0)
        )
    
    # Some medium workloads with potential for errors
    for i in range(4):
        varied_workloads.append(
            create_workload_spec(f"medium_job_{i}", priority=1, cpu_estimate=1.5, duration_estimate=4.0)
        )
    
    # Execute workloads in batches to create load patterns
    batch_size = 4
    for i in range(0, len(varied_workloads), batch_size):
        batch = varied_workloads[i:i + batch_size]
        print(f"   Executing batch {i//batch_size + 1} with {len(batch)} workloads...")
        
        await optimizer.execute_workload_batch(batch, max_concurrent=3)
        await asyncio.sleep(2)  # Brief pause between batches
    
    print("✅ Workload execution completed, analyzing performance data...")
    
    # Generate comprehensive performance report
    report = optimizer.create_optimization_report()
    
    print(f"\n🎯 PERFORMANCE OPTIMIZATION REPORT:")
    print("=" * 50)
    print(f"Overall Performance Score: {report['overall_performance_score']:.1f}/100")
    print(f"Health Status: {report['health_status'].upper()}")
    print(f"Report Generated: {report['timestamp'][:19]}")
    
    summary = report['summary']
    print(f"\n📈 SYSTEM SUMMARY:")
    print(f"   Total Instances: {summary['total_instances']}")
    print(f"   Healthy Instances: {summary['healthy_instances']}")
    print(f"   Pending Workloads: {summary['pending_workloads']}")
    print(f"   Running Workloads: {summary['running_workloads']}")
    print(f"   Identified Bottlenecks: {summary['identified_bottlenecks']}")
    
    # Show performance trends
    performance = report['detailed_status']['performance']
    trends = performance.get('trends', {})
    
    if trends:
        print(f"\n📊 PERFORMANCE TRENDS:")
        for operation, data in trends.items():
            print(f"   {operation}:")
            print(f"      Avg Latency: {data['avg_latency_ms']:.0f}ms")
            print(f"      Throughput: {data['avg_throughput_ops']:.2f} ops/sec")
            print(f"      Error Rate: {data['error_rate_percent']:.1f}%")
    
    # Show bottlenecks
    bottlenecks = performance.get('bottlenecks', [])
    if bottlenecks:
        print(f"\n🚨 IDENTIFIED BOTTLENECKS:")
        for bottleneck in bottlenecks:
            print(f"   {bottleneck['type'].upper()} in {bottleneck['operation']}")
            print(f"      Severity: {bottleneck['severity']}")
            print(f"      Value: {bottleneck['value']:.2f}")
            print(f"      Suggestion: {bottleneck['suggestion']}")
    else:
        print(f"\n✅ No significant bottlenecks detected")
    
    # Show optimization recommendations
    recommendations = performance.get('recommendations', [])
    if recommendations:
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Show next actions
    next_actions = report.get('next_actions', [])
    print(f"\n🎯 RECOMMENDED NEXT ACTIONS:")
    for i, action in enumerate(next_actions, 1):
        print(f"   {i}. {action}")
    
    await optimizer.stop_monitoring()


async def demo_comprehensive_scaling_scenario():
    """Demonstrate a comprehensive real-world scaling scenario."""
    print("\n🚀 === COMPREHENSIVE SCALING SCENARIO ===")
    print("Simulating a real-world quantum CI/CD pipeline with varying workloads\n")
    
    # Create optimizer with realistic settings
    optimizer = await create_quantum_optimizer(min_instances=2, max_instances=8, target_cpu=75.0)
    
    # Scenario: Morning batch processing, followed by interactive development, then evening optimization
    
    print("🌅 Phase 1: Morning Batch Processing (high sustained load)")
    
    # Morning: Large batch of quantum simulations for overnight results
    morning_batch = [
        create_workload_spec(f"batch_simulation_{i}", priority=2, cpu_estimate=2.0, duration_estimate=6.0)
        for i in range(12)
    ]
    
    print(f"   Submitting {len(morning_batch)} batch simulation jobs...")
    batch_start = time.time()
    
    # Submit all jobs quickly to simulate batch submission
    for workload in morning_batch:
        await optimizer.submit_quantum_workload(workload, OptimizationStrategy.THROUGHPUT)
    
    # Let some execute while monitoring scaling
    await asyncio.sleep(8)
    
    status = optimizer.get_comprehensive_status()
    print(f"   Current Instances: {status['auto_scaling']['current_instances']}")
    print(f"   Queue Depth: {status['scheduling']['pending_workloads']}")
    print(f"   CPU Utilization: {status['cluster']['cluster_utilization']['cpu']:.1f}%")
    
    print("\n👨‍💻 Phase 2: Interactive Development (mixed load)")
    
    # Daytime: Mixed interactive and development workloads
    interactive_workloads = [
        create_workload_spec(f"dev_test_{i}", priority=3, cpu_estimate=1.0, duration_estimate=2.0, deadline_minutes=10)
        for i in range(6)
    ] + [
        create_workload_spec(f"code_analysis_{i}", priority=1, cpu_estimate=0.5, duration_estimate=3.0)
        for i in range(4)
    ]
    
    print(f"   Processing {len(interactive_workloads)} interactive development workloads...")
    
    # Execute with lower concurrency to simulate interactive usage
    dev_results = await optimizer.execute_workload_batch(interactive_workloads, max_concurrent=2)
    dev_successful = sum(1 for r in dev_results if r.get("status") == "success")
    
    await asyncio.sleep(5)  # Allow scaling to adjust
    
    status = optimizer.get_comprehensive_status()
    print(f"   Completed: {dev_successful}/{len(interactive_workloads)} dev workloads")
    print(f"   Current Instances: {status['auto_scaling']['current_instances']}")
    
    print("\n🌙 Phase 3: Evening Optimization (compute-intensive)")
    
    # Evening: Resource-intensive optimization and analysis
    optimization_workloads = [
        create_workload_spec(f"circuit_optimization_{i}", priority=4, cpu_estimate=4.0, duration_estimate=10.0)
        for i in range(4)
    ] + [
        create_workload_spec(f"performance_analysis_{i}", priority=3, cpu_estimate=2.5, duration_estimate=7.0)
        for i in range(3)
    ]
    
    print(f"   Running {len(optimization_workloads)} intensive optimization jobs...")
    
    opt_results = await optimizer.execute_workload_batch(optimization_workloads, max_concurrent=3)
    opt_successful = sum(1 for r in opt_results if r.get("status") == "success")
    
    total_scenario_time = time.time() - batch_start
    
    print(f"   Completed: {opt_successful}/{len(optimization_workloads)} optimization workloads")
    
    # Final report
    final_report = optimizer.create_optimization_report()
    final_status = final_report['detailed_status']
    
    print(f"\n📋 COMPREHENSIVE SCENARIO RESULTS:")
    print("=" * 50)
    print(f"Total Scenario Time: {total_scenario_time:.2f}s")
    print(f"Overall Performance Score: {final_report['overall_performance_score']:.1f}/100")
    print(f"Health Status: {final_report['health_status'].upper()}")
    
    # Scaling efficiency analysis
    scaling_actions = final_status['auto_scaling']['recent_scaling_actions']
    scale_ups = len([a for a in scaling_actions if a['direction'] == 'UP'])
    scale_downs = len([a for a in scaling_actions if a['direction'] == 'DOWN'])
    
    print(f"\n⚖️ SCALING EFFICIENCY:")
    print(f"   Scale-up Actions: {scale_ups}")
    print(f"   Scale-down Actions: {scale_downs}")
    print(f"   Final Instance Count: {final_status['auto_scaling']['current_instances']}")
    print(f"   Peak Instances: {max(a['instances'] for a in scaling_actions) if scaling_actions else 'N/A'}")
    
    # Resource utilization summary
    cluster = final_status['cluster']
    print(f"\n💻 RESOURCE UTILIZATION:")
    print(f"   CPU Utilization: {cluster['cluster_utilization']['cpu']:.1f}%")
    print(f"   Memory Utilization: {cluster['cluster_utilization']['memory']:.1f}%")
    print(f"   Healthy Instances: {cluster['healthy_instances']}/{cluster['total_instances']}")
    
    await optimizer.stop_monitoring()


async def main():
    """Main demo function."""
    print("⚡ QUANTUM PERFORMANCE OPTIMIZER FOR SCALABLE DEVOPS CI/CD")
    print("=" * 70)
    print("Demonstrating advanced performance optimization features:")
    print("• Intelligent load balancing")
    print("• Dynamic auto-scaling")
    print("• Multiple optimization strategies")
    print("• Comprehensive performance monitoring")
    print("• Real-world scaling scenarios")
    
    try:
        # Run all demos
        await demo_load_balancing()
        await demo_auto_scaling()
        await demo_optimization_strategies()
        await demo_performance_monitoring()
        await demo_comprehensive_scaling_scenario()
        
        print("\n🎉 === QUANTUM PERFORMANCE OPTIMIZER DEMO COMPLETED ===")
        print("All performance optimization features demonstrated successfully!")
        print("System ready for production-scale quantum DevOps workloads.")
        
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