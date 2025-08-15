"""
Quantum Performance Optimizer for Scalable DevOps CI/CD.

Advanced performance optimization, auto-scaling, concurrent execution,
load balancing, and resource management for quantum computing workloads.
"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import heapq
import random

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    COST = "cost"
    ENERGY = "energy"
    BALANCED = "balanced"


class ScalingDirection(Enum):
    """Auto-scaling directions."""
    UP = auto()
    DOWN = auto()
    STABLE = auto()


class ResourceType(Enum):
    """Resource types for allocation."""
    CPU = "cpu"
    MEMORY = "memory"
    QUANTUM_BACKEND = "quantum_backend"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    throughput: float = 0.0  # operations per second
    latency: float = 0.0     # average response time in ms
    cpu_usage: float = 0.0   # CPU utilization percentage
    memory_usage: float = 0.0 # Memory utilization percentage
    error_rate: float = 0.0  # Error percentage
    cost_per_operation: float = 0.0  # Cost in USD
    energy_consumption: float = 0.0  # Energy in Joules
    queue_depth: int = 0     # Current queue size
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: float = 300.0  # seconds
    scale_down_cooldown: float = 600.0  # seconds
    scale_step_size: int = 1


@dataclass
class WorkloadSpec:
    """Workload specification for optimization."""
    workload_id: str
    priority: int = 1  # 1=low, 5=critical
    estimated_cpu: float = 1.0
    estimated_memory: float = 512.0  # MB
    estimated_duration: float = 60.0  # seconds
    deadline: Optional[datetime] = None
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    optimization_preferences: List[OptimizationStrategy] = field(default_factory=lambda: [OptimizationStrategy.BALANCED])
    
    def __lt__(self, other):
        """Make WorkloadSpec sortable for priority queue."""
        if not isinstance(other, WorkloadSpec):
            return NotImplemented
        return self.workload_id < other.workload_id


@dataclass
class ResourceAllocation:
    """Resource allocation result."""
    allocated_resources: Dict[ResourceType, float]
    instance_assignment: str
    estimated_completion_time: datetime
    cost_estimate: float
    confidence_score: float


class LoadBalancer:
    """Intelligent load balancer for quantum workloads."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.load_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.routing_algorithm = "weighted_round_robin"
        
    def register_instance(self, instance_id: str, capacity: Dict[str, float]):
        """Register a compute instance."""
        self.instances[instance_id] = {
            "capacity": capacity,
            "current_load": {resource: 0.0 for resource in capacity.keys()},
            "health_score": 1.0,
            "last_updated": datetime.now(),
            "total_requests": 0,
            "failed_requests": 0,
            "response_times": deque(maxlen=100)
        }
        logger.info(f"Registered instance {instance_id} with capacity {capacity}")
    
    def update_instance_metrics(self, instance_id: str, metrics: PerformanceMetrics):
        """Update instance performance metrics."""
        if instance_id not in self.instances:
            logger.warning(f"Unknown instance {instance_id}")
            return
        
        instance = self.instances[instance_id]
        instance["current_load"]["cpu"] = metrics.cpu_usage
        instance["current_load"]["memory"] = metrics.memory_usage
        instance["last_updated"] = datetime.now()
        
        # Update health score based on error rate and response time
        error_penalty = metrics.error_rate * 0.5
        latency_penalty = min(metrics.latency / 1000.0, 0.3)  # Cap at 30%
        instance["health_score"] = max(0.1, 1.0 - error_penalty - latency_penalty)
        
        # Store metrics for analysis
        self.load_metrics[instance_id].append(metrics)
    
    def select_instance(self, workload: WorkloadSpec) -> Optional[str]:
        """Select optimal instance for workload."""
        if not self.instances:
            return None
        
        # Filter instances that can handle the workload
        viable_instances = []
        
        for instance_id, instance_data in self.instances.items():
            # Check capacity constraints
            cpu_available = instance_data["capacity"]["cpu"] - instance_data["current_load"]["cpu"]
            memory_available = instance_data["capacity"]["memory"] - instance_data["current_load"]["memory"]
            
            if (cpu_available >= workload.estimated_cpu and 
                memory_available >= workload.estimated_memory and
                instance_data["health_score"] > 0.5):
                
                # Calculate score based on available capacity and health
                capacity_score = (cpu_available / instance_data["capacity"]["cpu"] + 
                                memory_available / instance_data["capacity"]["memory"]) / 2
                
                total_score = (capacity_score * 0.7 + instance_data["health_score"] * 0.3)
                viable_instances.append((instance_id, total_score))
        
        if not viable_instances:
            logger.warning("No viable instances found for workload")
            return None
        
        # Select instance with highest score
        viable_instances.sort(key=lambda x: x[1], reverse=True)
        selected = viable_instances[0][0]
        
        # Update load tracking
        self.instances[selected]["current_load"]["cpu"] += workload.estimated_cpu
        self.instances[selected]["current_load"]["memory"] += workload.estimated_memory
        self.instances[selected]["total_requests"] += 1
        
        return selected
    
    def release_resources(self, instance_id: str, workload: WorkloadSpec):
        """Release resources after workload completion."""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        instance["current_load"]["cpu"] = max(0, 
            instance["current_load"]["cpu"] - workload.estimated_cpu)
        instance["current_load"]["memory"] = max(0,
            instance["current_load"]["memory"] - workload.estimated_memory)
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get cluster-wide performance metrics."""
        if not self.instances:
            return {"status": "no_instances"}
        
        total_capacity = {"cpu": 0.0, "memory": 0.0}
        total_load = {"cpu": 0.0, "memory": 0.0}
        healthy_instances = 0
        
        for instance_data in self.instances.values():
            for resource in ["cpu", "memory"]:
                total_capacity[resource] += instance_data["capacity"][resource]
                total_load[resource] += instance_data["current_load"][resource]
            
            if instance_data["health_score"] > 0.7:
                healthy_instances += 1
        
        utilization = {}
        for resource in ["cpu", "memory"]:
            utilization[resource] = (total_load[resource] / total_capacity[resource] * 100
                                   if total_capacity[resource] > 0 else 0)
        
        return {
            "total_instances": len(self.instances),
            "healthy_instances": healthy_instances,
            "cluster_utilization": utilization,
            "total_capacity": total_capacity,
            "total_load": total_load
        }


class AutoScaler:
    """Auto-scaling manager for dynamic resource allocation."""
    
    def __init__(self, config: ScalingConfig, load_balancer: LoadBalancer):
        """Initialize auto-scaler."""
        self.config = config
        self.load_balancer = load_balancer
        self.current_instances = 1
        self.last_scale_action = datetime.now()
        self.scaling_history: List[Tuple[datetime, ScalingDirection, int]] = []
        self.metrics_history: deque = deque(maxlen=50)
        
    async def evaluate_scaling_decision(self, current_metrics: PerformanceMetrics) -> ScalingDirection:
        """Evaluate whether scaling action is needed."""
        self.metrics_history.append(current_metrics)
        
        # Need sufficient history for decision
        if len(self.metrics_history) < 5:
            return ScalingDirection.STABLE
        
        # Calculate recent averages
        recent_metrics = list(self.metrics_history)[-5:]
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_queue_depth = sum(m.queue_depth for m in recent_metrics) / len(recent_metrics)
        
        # Check cooldown periods
        now = datetime.now()
        time_since_last_scale = (now - self.last_scale_action).total_seconds()
        
        # Scale up conditions
        if (avg_cpu > self.config.scale_up_threshold or 
            avg_memory > self.config.scale_up_threshold or
            avg_queue_depth > 10):
            
            if (self.current_instances < self.config.max_instances and
                time_since_last_scale > self.config.scale_up_cooldown):
                return ScalingDirection.UP
        
        # Scale down conditions
        elif (avg_cpu < self.config.scale_down_threshold and 
              avg_memory < self.config.scale_down_threshold and
              avg_queue_depth < 2):
            
            if (self.current_instances > self.config.min_instances and
                time_since_last_scale > self.config.scale_down_cooldown):
                return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    async def execute_scaling_action(self, direction: ScalingDirection) -> bool:
        """Execute scaling action."""
        if direction == ScalingDirection.STABLE:
            return True
        
        old_instances = self.current_instances
        
        if direction == ScalingDirection.UP:
            new_instances = min(
                self.current_instances + self.config.scale_step_size,
                self.config.max_instances
            )
            
            # Add new instances
            for i in range(self.current_instances, new_instances):
                instance_id = f"quantum_worker_{i}"
                capacity = {"cpu": 8.0, "memory": 16384.0}  # 8 CPU, 16GB RAM
                self.load_balancer.register_instance(instance_id, capacity)
            
            logger.info(f"Scaled up from {old_instances} to {new_instances} instances")
            
        elif direction == ScalingDirection.DOWN:
            new_instances = max(
                self.current_instances - self.config.scale_step_size,
                self.config.min_instances
            )
            
            # Remove instances (in production, would gracefully drain workloads)
            instances_to_remove = []
            instance_ids = list(self.load_balancer.instances.keys())
            
            for i in range(new_instances, self.current_instances):
                if i < len(instance_ids):
                    instance_id = instance_ids[i]
                    instances_to_remove.append(instance_id)
            
            for instance_id in instances_to_remove:
                del self.load_balancer.instances[instance_id]
            
            logger.info(f"Scaled down from {old_instances} to {new_instances} instances")
        
        self.current_instances = new_instances
        self.last_scale_action = datetime.now()
        self.scaling_history.append((datetime.now(), direction, new_instances))
        
        return True
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get auto-scaling metrics and history."""
        recent_scaling = [
            {
                "timestamp": ts.isoformat(),
                "direction": direction.name,
                "instances": count
            }
            for ts, direction, count in self.scaling_history[-10:]
        ]
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.config.min_instances,
            "max_instances": self.config.max_instances,
            "last_scale_action": self.last_scale_action.isoformat(),
            "recent_scaling_actions": recent_scaling,
            "metrics_buffer_size": len(self.metrics_history)
        }


class QuantumWorkloadScheduler:
    """Advanced scheduler for quantum computing workloads."""
    
    def __init__(self, load_balancer: LoadBalancer):
        """Initialize quantum workload scheduler."""
        self.load_balancer = load_balancer
        self.pending_queue: List[Tuple[float, WorkloadSpec]] = []  # priority heap
        self.running_workloads: Dict[str, WorkloadSpec] = {}
        self.completed_workloads: List[WorkloadSpec] = []
        self.scheduling_algorithm = "priority_deadline"
        
    def submit_workload(self, workload: WorkloadSpec) -> str:
        """Submit workload for scheduling."""
        # Calculate priority score (higher score = higher priority)
        priority_score = workload.priority * 1000  # Base priority
        
        # Add deadline urgency
        if workload.deadline:
            time_to_deadline = (workload.deadline - datetime.now()).total_seconds()
            if time_to_deadline > 0:
                urgency_bonus = max(0, 1000 - time_to_deadline / 60)  # Bonus for tight deadlines
                priority_score += urgency_bonus
        
        # Add to priority queue (negative for max-heap behavior)
        heapq.heappush(self.pending_queue, (-priority_score, workload))
        
        logger.info(f"Submitted workload {workload.workload_id} with priority {priority_score}")
        return workload.workload_id
    
    async def schedule_next_workload(self) -> Optional[Tuple[str, WorkloadSpec]]:
        """Schedule the next highest-priority workload."""
        if not self.pending_queue:
            return None
        
        # Get highest priority workload
        priority_score, workload = heapq.heappop(self.pending_queue)
        
        # Select instance for execution
        selected_instance = self.load_balancer.select_instance(workload)
        
        if selected_instance is None:
            # No available instances, put back in queue
            heapq.heappush(self.pending_queue, (priority_score, workload))
            return None
        
        # Mark as running
        self.running_workloads[workload.workload_id] = workload
        
        logger.info(f"Scheduled workload {workload.workload_id} on instance {selected_instance}")
        return selected_instance, workload
    
    def complete_workload(self, workload_id: str, success: bool = True) -> bool:
        """Mark workload as completed."""
        if workload_id not in self.running_workloads:
            return False
        
        workload = self.running_workloads.pop(workload_id)
        self.completed_workloads.append(workload)
        
        # Release resources
        # Note: In production, we'd track which instance ran this workload
        logger.info(f"Completed workload {workload_id}, success: {success}")
        return True
    
    def get_queue_metrics(self) -> Dict[str, Any]:
        """Get scheduling queue metrics."""
        return {
            "pending_workloads": len(self.pending_queue),
            "running_workloads": len(self.running_workloads),
            "completed_workloads": len(self.completed_workloads),
            "queue_by_priority": self._analyze_queue_priorities()
        }
    
    def _analyze_queue_priorities(self) -> Dict[str, int]:
        """Analyze queue composition by priority."""
        priority_counts = defaultdict(int)
        for priority_score, workload in self.pending_queue:
            priority_counts[f"priority_{workload.priority}"] += 1
        return dict(priority_counts)


class PerformanceProfiler:
    """Performance profiler for quantum operations."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.operation_profiles: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.bottleneck_analysis: Dict[str, Dict[str, float]] = {}
        self.optimization_suggestions: List[str] = []
        
    def record_operation(self, operation_type: str, metrics: PerformanceMetrics):
        """Record performance metrics for an operation."""
        self.operation_profiles[operation_type].append(metrics)
        
        # Keep only recent data
        if len(self.operation_profiles[operation_type]) > 100:
            self.operation_profiles[operation_type] = \
                self.operation_profiles[operation_type][-100:]
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and identify optimization opportunities."""
        analysis = {}
        
        for operation_type, metrics_list in self.operation_profiles.items():
            if len(metrics_list) < 5:
                continue
            
            # Calculate averages and trends
            recent_metrics = metrics_list[-10:]
            older_metrics = metrics_list[-20:-10] if len(metrics_list) >= 20 else []
            
            avg_latency = sum(m.latency for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            
            # Trend analysis
            trends = {}
            if older_metrics:
                old_avg_latency = sum(m.latency for m in older_metrics) / len(older_metrics)
                old_avg_throughput = sum(m.throughput for m in older_metrics) / len(older_metrics)
                
                trends["latency_trend"] = (avg_latency - old_avg_latency) / old_avg_latency if old_avg_latency > 0 else 0
                trends["throughput_trend"] = (avg_throughput - old_avg_throughput) / old_avg_throughput if old_avg_throughput > 0 else 0
            
            analysis[operation_type] = {
                "avg_latency_ms": avg_latency,
                "avg_throughput_ops": avg_throughput,
                "error_rate_percent": avg_error_rate * 100,
                "sample_count": len(recent_metrics),
                "trends": trends
            }
        
        return analysis
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for operation_type, metrics_list in self.operation_profiles.items():
            if len(metrics_list) < 5:
                continue
            
            recent_metrics = metrics_list[-10:]
            
            # High latency bottleneck
            avg_latency = sum(m.latency for m in recent_metrics) / len(recent_metrics)
            if avg_latency > 5000:  # 5 seconds
                bottlenecks.append({
                    "type": "high_latency",
                    "operation": operation_type,
                    "severity": "high" if avg_latency > 10000 else "medium",
                    "value": avg_latency,
                    "suggestion": "Consider optimizing algorithm or adding more compute resources"
                })
            
            # High error rate bottleneck
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            if avg_error_rate > 0.05:  # 5% error rate
                bottlenecks.append({
                    "type": "high_error_rate",
                    "operation": operation_type,
                    "severity": "high" if avg_error_rate > 0.1 else "medium",
                    "value": avg_error_rate * 100,
                    "suggestion": "Review error handling and input validation"
                })
            
            # Low throughput bottleneck
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            if avg_throughput < 1.0:  # Less than 1 op/sec
                bottlenecks.append({
                    "type": "low_throughput",
                    "operation": operation_type,
                    "severity": "medium",
                    "value": avg_throughput,
                    "suggestion": "Consider parallel processing or resource scaling"
                })
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        bottlenecks = self.identify_bottlenecks()
        
        # Group bottlenecks by type
        bottleneck_types = defaultdict(list)
        for bottleneck in bottlenecks:
            bottleneck_types[bottleneck["type"]].append(bottleneck)
        
        # Generate recommendations
        if "high_latency" in bottleneck_types:
            recommendations.append(
                "High latency detected in multiple operations. "
                "Consider implementing caching, circuit optimization, or adding compute resources."
            )
        
        if "high_error_rate" in bottleneck_types:
            recommendations.append(
                "High error rates detected. "
                "Review input validation, error handling, and consider implementing circuit verification."
            )
        
        if "low_throughput" in bottleneck_types:
            recommendations.append(
                "Low throughput detected. "
                "Consider implementing parallel processing, batch operations, or auto-scaling."
            )
        
        return recommendations


class QuantumPerformanceOptimizer:
    """
    Comprehensive performance optimizer for quantum DevOps CI/CD.
    
    Features:
    - Intelligent load balancing across quantum backends
    - Auto-scaling based on workload demands
    - Advanced scheduling with priority and deadline awareness
    - Performance profiling and bottleneck identification
    - Resource optimization recommendations
    """
    
    def __init__(self, scaling_config: Optional[ScalingConfig] = None):
        """Initialize quantum performance optimizer."""
        self.load_balancer = LoadBalancer()
        self.scaling_config = scaling_config or ScalingConfig()
        self.auto_scaler = AutoScaler(self.scaling_config, self.load_balancer)
        self.scheduler = QuantumWorkloadScheduler(self.load_balancer)
        self.profiler = PerformanceProfiler()
        
        # Performance monitoring
        self.metrics_history: deque = deque(maxlen=1000)
        self.optimization_enabled = True
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize default instances
        self._initialize_default_instances()
    
    def _initialize_default_instances(self):
        """Initialize default compute instances."""
        # Simulate different types of quantum computing instances
        instances = [
            ("quantum_simulator_1", {"cpu": 4.0, "memory": 8192.0}),
            ("quantum_simulator_2", {"cpu": 8.0, "memory": 16384.0}),
            ("quantum_hardware_1", {"cpu": 2.0, "memory": 4096.0}),
        ]
        
        for instance_id, capacity in instances:
            self.load_balancer.register_instance(instance_id, capacity)
    
    async def submit_quantum_workload(
        self, 
        workload_spec: WorkloadSpec,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> str:
        """Submit quantum workload for optimized execution."""
        
        # Apply optimization based on strategy
        optimized_spec = await self._optimize_workload_spec(workload_spec, optimization_strategy)
        
        # Submit to scheduler
        workload_id = self.scheduler.submit_workload(optimized_spec)
        
        logger.info(f"Submitted quantum workload {workload_id} with {optimization_strategy.value} optimization")
        return workload_id
    
    async def _optimize_workload_spec(
        self, 
        workload: WorkloadSpec, 
        strategy: OptimizationStrategy
    ) -> WorkloadSpec:
        """Optimize workload specification based on strategy."""
        
        optimized = WorkloadSpec(
            workload_id=workload.workload_id,
            priority=workload.priority,
            estimated_cpu=workload.estimated_cpu,
            estimated_memory=workload.estimated_memory,
            estimated_duration=workload.estimated_duration,
            deadline=workload.deadline,
            resource_requirements=workload.resource_requirements.copy(),
            optimization_preferences=[strategy]
        )
        
        if strategy == OptimizationStrategy.THROUGHPUT:
            # Optimize for maximum throughput
            optimized.estimated_cpu *= 1.5  # Use more CPU for parallel processing
            optimized.priority = min(5, optimized.priority + 1)  # Increase priority
            
        elif strategy == OptimizationStrategy.LATENCY:
            # Optimize for minimum latency
            optimized.estimated_cpu *= 2.0  # Dedicate more resources
            optimized.priority = 5  # Highest priority
            
        elif strategy == OptimizationStrategy.COST:
            # Optimize for minimum cost
            optimized.estimated_cpu *= 0.7  # Use fewer resources
            optimized.estimated_duration *= 1.3  # Allow longer execution time
            if not optimized.deadline:
                # Set relaxed deadline for cost optimization
                optimized.deadline = datetime.now() + timedelta(hours=2)
                
        elif strategy == OptimizationStrategy.ENERGY:
            # Optimize for energy efficiency
            optimized.estimated_cpu *= 0.8  # Reduce CPU usage
            optimized.estimated_duration *= 1.2  # Allow longer execution
            
        return optimized
    
    async def execute_workload_batch(
        self, 
        workloads: List[WorkloadSpec],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute multiple workloads with optimal concurrency."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_workload(workload: WorkloadSpec) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                
                try:
                    # Submit and wait for scheduling
                    workload_id = await self.submit_quantum_workload(workload)
                    
                    # Simulate workload execution
                    await asyncio.sleep(workload.estimated_duration / 10.0)  # Scaled for demo
                    
                    execution_time = time.time() - start_time
                    
                    # Record metrics
                    metrics = PerformanceMetrics(
                        throughput=1.0 / execution_time if execution_time > 0 else 0,
                        latency=execution_time * 1000,  # Convert to ms
                        cpu_usage=random.uniform(30, 90),
                        memory_usage=random.uniform(40, 80),
                        error_rate=random.uniform(0, 0.05),
                        cost_per_operation=workload.estimated_cpu * 0.1,
                        energy_consumption=workload.estimated_cpu * workload.estimated_duration * 0.5
                    )
                    
                    self.profiler.record_operation("quantum_execution", metrics)
                    
                    # Mark as completed
                    self.scheduler.complete_workload(workload_id, success=True)
                    
                    return {
                        "workload_id": workload_id,
                        "status": "success",
                        "execution_time": execution_time,
                        "metrics": metrics
                    }
                    
                except Exception as e:
                    logger.error(f"Workload execution failed: {e}")
                    return {
                        "workload_id": workload.workload_id,
                        "status": "failed",
                        "error": str(e),
                        "execution_time": time.time() - start_time
                    }
        
        # Execute all workloads concurrently
        tasks = [execute_single_workload(workload) for workload in workloads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "exception",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start performance monitoring and auto-scaling."""
        if self.monitoring_task:
            return  # Already running
        
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )
        logger.info("Started performance monitoring and auto-scaling")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("Stopped performance monitoring")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while True:
            try:
                # Collect current metrics
                cluster_metrics = self.load_balancer.get_cluster_metrics()
                queue_metrics = self.scheduler.get_queue_metrics()
                
                # Create aggregated performance metrics
                current_metrics = PerformanceMetrics(
                    cpu_usage=cluster_metrics.get("cluster_utilization", {}).get("cpu", 0),
                    memory_usage=cluster_metrics.get("cluster_utilization", {}).get("memory", 0),
                    queue_depth=queue_metrics.get("pending_workloads", 0),
                    timestamp=datetime.now()
                )
                
                self.metrics_history.append(current_metrics)
                
                # Evaluate scaling decision
                scaling_decision = await self.auto_scaler.evaluate_scaling_decision(current_metrics)
                
                # Execute scaling if needed
                if scaling_decision != ScalingDirection.STABLE:
                    await self.auto_scaler.execute_scaling_action(scaling_decision)
                
                # Update optimization suggestions
                if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                    await self._update_optimization_suggestions()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _update_optimization_suggestions(self):
        """Update optimization suggestions based on performance analysis."""
        try:
            recommendations = self.profiler.generate_optimization_recommendations()
            if recommendations:
                logger.info(f"New optimization recommendations: {recommendations}")
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive performance optimization status."""
        
        # Cluster status
        cluster_metrics = self.load_balancer.get_cluster_metrics()
        
        # Queue status
        queue_metrics = self.scheduler.get_queue_metrics()
        
        # Scaling status
        scaling_metrics = self.auto_scaler.get_scaling_metrics()
        
        # Performance analysis
        performance_trends = self.profiler.analyze_performance_trends()
        bottlenecks = self.profiler.identify_bottlenecks()
        recommendations = self.profiler.generate_optimization_recommendations()
        
        # Recent metrics summary
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_queue_depth = sum(m.queue_depth for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = avg_memory = avg_queue_depth = 0
        
        return {
            "cluster": cluster_metrics,
            "scheduling": queue_metrics,
            "auto_scaling": scaling_metrics,
            "performance": {
                "trends": performance_trends,
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "recent_averages": {
                    "cpu_utilization": avg_cpu,
                    "memory_utilization": avg_memory,
                    "queue_depth": avg_queue_depth
                }
            },
            "monitoring": {
                "active": self.monitoring_task is not None and not self.monitoring_task.done(),
                "metrics_collected": len(self.metrics_history)
            }
        }
    
    def create_optimization_report(self) -> Dict[str, Any]:
        """Create detailed optimization report."""
        status = self.get_comprehensive_status()
        
        # Performance score calculation
        cluster = status["cluster"]
        performance = status["performance"]
        
        # Calculate overall performance score (0-100)
        cpu_score = max(0, 100 - cluster.get("cluster_utilization", {}).get("cpu", 0))
        memory_score = max(0, 100 - cluster.get("cluster_utilization", {}).get("memory", 0))
        queue_score = max(0, 100 - status["scheduling"].get("pending_workloads", 0) * 5)
        bottleneck_penalty = len(performance.get("bottlenecks", [])) * 10
        
        overall_score = max(0, (cpu_score + memory_score + queue_score) / 3 - bottleneck_penalty)
        
        # Determine health status
        if overall_score >= 80:
            health_status = "excellent"
        elif overall_score >= 60:
            health_status = "good"
        elif overall_score >= 40:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_performance_score": overall_score,
            "health_status": health_status,
            "summary": {
                "total_instances": cluster.get("total_instances", 0),
                "healthy_instances": cluster.get("healthy_instances", 0),
                "pending_workloads": status["scheduling"].get("pending_workloads", 0),
                "running_workloads": status["scheduling"].get("running_workloads", 0),
                "identified_bottlenecks": len(performance.get("bottlenecks", [])),
                "optimization_recommendations": len(performance.get("recommendations", []))
            },
            "detailed_status": status,
            "next_actions": self._generate_next_actions(status, overall_score)
        }
    
    def _generate_next_actions(self, status: Dict[str, Any], performance_score: float) -> List[str]:
        """Generate recommended next actions based on current status."""
        actions = []
        
        cluster = status["cluster"]
        performance = status["performance"]
        
        # High utilization
        cpu_util = cluster.get("cluster_utilization", {}).get("cpu", 0)
        if cpu_util > 80:
            actions.append("Consider scaling up - high CPU utilization detected")
        
        # Queue buildup
        pending = status["scheduling"].get("pending_workloads", 0)
        if pending > 10:
            actions.append("Consider adding more instances - large queue detected")
        
        # Bottlenecks
        bottlenecks = performance.get("bottlenecks", [])
        if bottlenecks:
            high_severity = [b for b in bottlenecks if b.get("severity") == "high"]
            if high_severity:
                actions.append("Address high-severity bottlenecks immediately")
            else:
                actions.append("Review and address identified performance bottlenecks")
        
        # Low performance score
        if performance_score < 50:
            actions.append("Comprehensive performance review recommended")
        
        # No immediate issues
        if not actions:
            actions.append("System operating within normal parameters")
        
        return actions


# Convenience functions for easy integration
async def create_quantum_optimizer(
    min_instances: int = 1,
    max_instances: int = 10,
    target_cpu: float = 70.0
) -> QuantumPerformanceOptimizer:
    """Create a fully configured quantum performance optimizer."""
    
    scaling_config = ScalingConfig(
        min_instances=min_instances,
        max_instances=max_instances,
        target_cpu_utilization=target_cpu,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0
    )
    
    optimizer = QuantumPerformanceOptimizer(scaling_config)
    
    # Start monitoring
    await optimizer.start_monitoring(interval=10.0)  # Monitor every 10 seconds
    
    logger.info("Quantum performance optimizer created and monitoring started")
    return optimizer


def create_workload_spec(
    workload_id: str,
    priority: int = 1,
    cpu_estimate: float = 1.0,
    memory_estimate: float = 512.0,
    duration_estimate: float = 60.0,
    deadline_minutes: Optional[int] = None
) -> WorkloadSpec:
    """Create a workload specification with sensible defaults."""
    
    deadline = None
    if deadline_minutes:
        deadline = datetime.now() + timedelta(minutes=deadline_minutes)
    
    return WorkloadSpec(
        workload_id=workload_id,
        priority=priority,
        estimated_cpu=cpu_estimate,
        estimated_memory=memory_estimate,
        estimated_duration=duration_estimate,
        deadline=deadline
    )