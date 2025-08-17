"""
Adaptive load balancing and auto-scaling for quantum computing workloads.
Generation 3 implementation with intelligent resource management.
"""

import asyncio
import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import heapq
import json

from .exceptions import QuantumResourceError, QuantumTimeoutError
from .monitoring import ResourceMonitor

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies for quantum workloads."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    QUANTUM_OPTIMIZED = "quantum_optimized"

class BackendStatus(Enum):
    """Quantum backend status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"

@dataclass
class QuantumBackend:
    """Quantum backend representation for load balancing."""
    id: str
    name: str
    backend_type: str  # simulator, hardware
    max_qubits: int
    max_concurrent_jobs: int
    current_load: int = 0
    status: BackendStatus = BackendStatus.HEALTHY
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    cost_per_shot: float = 0.0
    queue_length: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_health_check: float = 0.0
    weight: float = 1.0

@dataclass
class WorkloadRequest:
    """Quantum workload request for load balancing."""
    id: str
    circuit: Any
    shots: int
    priority: int = 5  # 1-10, higher is more important
    max_qubits: int = 0
    estimated_runtime: float = 0.0
    deadline: Optional[float] = None
    cost_budget: Optional[float] = None
    backend_preferences: List[str] = field(default_factory=list)
    retry_count: int = 0

class AdaptiveLoadBalancer:
    """
    Intelligent load balancer for quantum computing workloads with auto-scaling.
    """
    
    def __init__(self, balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.QUANTUM_OPTIMIZED):
        self.strategy = balancing_strategy
        
        # Backend registry
        self.backends: Dict[str, QuantumBackend] = {}
        self.backend_pools: Dict[str, List[str]] = defaultdict(list)
        
        # Request queues
        self.pending_requests = queue.PriorityQueue()
        self.active_requests: Dict[str, WorkloadRequest] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.connection_counts: Dict[str, int] = defaultdict(int)
        
        # Performance monitoring
        self.resource_monitor = ResourceMonitor()
        self.performance_history = defaultdict(deque)
        self.health_check_interval = 30  # seconds
        
        # Auto-scaling configuration
        self.scaling_config = {
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.3,
            'min_backends': 2,
            'max_backends': 10,
            'cooldown_period': 300  # 5 minutes
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring and health check tasks."""
        
        # Health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        # Load balancing optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        # Auto-scaling thread
        self.scaling_thread = threading.Thread(
            target=self._auto_scaling_loop,
            daemon=True
        )
        self.scaling_thread.start()
    
    def register_backend(self, backend: QuantumBackend) -> None:
        """Register a quantum backend with the load balancer."""
        self.backends[backend.id] = backend
        self.backend_pools[backend.backend_type].append(backend.id)
        self.connection_counts[backend.id] = 0
        
        logger.info(f"Registered backend: {backend.name} ({backend.id})")
    
    def unregister_backend(self, backend_id: str) -> None:
        """Unregister a quantum backend."""
        if backend_id in self.backends:
            backend = self.backends[backend_id]
            self.backend_pools[backend.backend_type].remove(backend_id)
            del self.backends[backend_id]
            del self.connection_counts[backend_id]
            
            logger.info(f"Unregistered backend: {backend_id}")
    
    async def submit_request(self, request: WorkloadRequest) -> str:
        """Submit a quantum workload request for execution."""
        
        # Analyze request requirements
        self._analyze_request(request)
        
        # Find suitable backends
        suitable_backends = self._find_suitable_backends(request)
        
        if not suitable_backends:
            raise QuantumResourceError(f"No suitable backends available for request {request.id}")
        
        # Select optimal backend
        selected_backend = self._select_backend(suitable_backends, request)
        
        # Submit to backend
        backend_id = await self._submit_to_backend(selected_backend, request)
        
        # Track active request
        self.active_requests[request.id] = request
        self.connection_counts[selected_backend] += 1
        
        logger.info(f"Submitted request {request.id} to backend {selected_backend}")
        
        return backend_id
    
    def _analyze_request(self, request: WorkloadRequest) -> None:
        """Analyze request requirements and characteristics."""
        
        try:
            # Extract circuit properties
            if hasattr(request.circuit, 'num_qubits'):
                request.max_qubits = request.circuit.num_qubits
            
            if hasattr(request.circuit, 'depth'):
                circuit_depth = request.circuit.depth()
                # Estimate runtime based on depth and shots
                request.estimated_runtime = max(0.1, circuit_depth * request.shots * 0.001)
            
            # Set deadline if not provided
            if request.deadline is None:
                request.deadline = time.time() + max(300, request.estimated_runtime * 2)
                
        except Exception as e:
            logger.warning(f"Request analysis failed: {e}")
            request.estimated_runtime = 60.0  # Default fallback
    
    def _find_suitable_backends(self, request: WorkloadRequest) -> List[str]:
        """Find backends that can handle the request."""
        suitable = []
        
        for backend_id, backend in self.backends.items():
            # Check basic requirements
            if backend.status == BackendStatus.FAILED:
                continue
            
            if backend.max_qubits < request.max_qubits:
                continue
            
            if backend.current_load >= backend.max_concurrent_jobs:
                continue
            
            # Check preferences
            if request.backend_preferences:
                if backend_id not in request.backend_preferences and backend.name not in request.backend_preferences:
                    continue
            
            # Check cost constraints
            if request.cost_budget is not None:
                estimated_cost = backend.cost_per_shot * request.shots
                if estimated_cost > request.cost_budget:
                    continue
            
            suitable.append(backend_id)
        
        return suitable
    
    def _select_backend(self, suitable_backends: List[str], request: WorkloadRequest) -> str:
        """Select the optimal backend using the configured strategy."""
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(suitable_backends)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(suitable_backends)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(suitable_backends)
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(suitable_backends, request)
        
        elif self.strategy == LoadBalancingStrategy.QUANTUM_OPTIMIZED:
            return self._quantum_optimized_selection(suitable_backends, request)
        
        else:
            return suitable_backends[0]  # Fallback
    
    def _round_robin_selection(self, backends: List[str]) -> str:
        """Simple round-robin selection."""
        selected = backends[self.round_robin_index % len(backends)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, backends: List[str]) -> str:
        """Select backend with least active connections."""
        return min(backends, key=lambda b: self.connection_counts[b])
    
    def _weighted_round_robin_selection(self, backends: List[str]) -> str:
        """Weighted round-robin based on backend capacity."""
        weights = [self.backends[b].weight for b in backends]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return backends[0]
        
        # Weighted random selection
        r = np.random.random() * total_weight
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return backends[i]
        
        return backends[-1]
    
    def _resource_aware_selection(self, backends: List[str], request: WorkloadRequest) -> str:
        """Select backend based on current resource utilization."""
        
        def score_backend(backend_id: str) -> float:
            backend = self.backends[backend_id]
            
            # Calculate utilization score (lower is better)
            load_ratio = backend.current_load / max(1, backend.max_concurrent_jobs)
            queue_penalty = backend.queue_length * 0.1
            response_penalty = backend.response_time_ms / 1000.0
            
            # Success rate bonus (higher is better)
            reliability_bonus = backend.success_rate
            
            score = load_ratio + queue_penalty + response_penalty - reliability_bonus
            return score
        
        return min(backends, key=score_backend)
    
    def _quantum_optimized_selection(self, backends: List[str], request: WorkloadRequest) -> str:
        """Quantum-optimized selection considering circuit characteristics."""
        
        def quantum_score(backend_id: str) -> float:
            backend = self.backends[backend_id]
            score = 0.0
            
            # Hardware backends preferred for large circuits
            if request.max_qubits >= 10 and backend.backend_type == "hardware":
                score += 2.0
            
            # Simulator efficiency for small circuits
            if request.max_qubits < 10 and backend.backend_type == "simulator":
                score += 1.0
            
            # Qubit efficiency score
            qubit_efficiency = 1.0 - abs(backend.max_qubits - request.max_qubits) / backend.max_qubits
            score += qubit_efficiency
            
            # Cost optimization
            if request.cost_budget:
                cost_ratio = (backend.cost_per_shot * request.shots) / request.cost_budget
                score += (1.0 - min(1.0, cost_ratio))
            
            # Current load penalty
            load_penalty = backend.current_load / max(1, backend.max_concurrent_jobs)
            score -= load_penalty * 2.0
            
            # Response time consideration
            response_penalty = min(2.0, backend.response_time_ms / 1000.0)
            score -= response_penalty
            
            # Success rate bonus
            score += backend.success_rate * 1.5
            
            return score
        
        return max(backends, key=quantum_score)
    
    async def _submit_to_backend(self, backend_id: str, request: WorkloadRequest) -> str:
        """Submit request to the selected backend."""
        
        backend = self.backends[backend_id]
        
        try:
            # Update backend load
            backend.current_load += 1
            backend.queue_length += 1
            
            # Simulate backend submission (in real implementation, this would call the actual backend)
            submission_time = time.time()
            
            # Record submission
            self.performance_history[backend_id].append({
                'timestamp': submission_time,
                'request_id': request.id,
                'estimated_runtime': request.estimated_runtime,
                'qubits': request.max_qubits,
                'shots': request.shots
            })
            
            return backend_id
            
        except Exception as e:
            # Rollback on failure
            backend.current_load = max(0, backend.current_load - 1)
            backend.queue_length = max(0, backend.queue_length - 1)
            raise QuantumResourceError(f"Failed to submit to backend {backend_id}: {e}")
    
    def complete_request(self, request_id: str, backend_id: str, success: bool = True) -> None:
        """Mark a request as completed and update backend statistics."""
        
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            backend = self.backends[backend_id]
            
            # Update backend state
            backend.current_load = max(0, backend.current_load - 1)
            backend.queue_length = max(0, backend.queue_length - 1)
            self.connection_counts[backend_id] = max(0, self.connection_counts[backend_id] - 1)
            
            # Update success rate
            history = self.performance_history[backend_id]
            if len(history) > 0:
                recent_success_rate = np.mean([1.0 if 'success' not in h else h['success'] for h in list(history)[-10:]])
                backend.success_rate = 0.9 * backend.success_rate + 0.1 * (1.0 if success else 0.0)
            
            # Remove from active requests
            del self.active_requests[request_id]
            
            logger.info(f"Completed request {request_id} on backend {backend_id} (success: {success})")
    
    def _health_check_loop(self) -> None:
        """Background health checking for all backends."""
        
        while True:
            try:
                current_time = time.time()
                
                for backend_id, backend in self.backends.items():
                    if current_time - backend.last_health_check > self.health_check_interval:
                        self._check_backend_health(backend)
                        backend.last_health_check = current_time
                
                time.sleep(min(10, self.health_check_interval / 2))
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(10)
    
    def _check_backend_health(self, backend: QuantumBackend) -> None:
        """Check health of a specific backend."""
        
        try:
            # Simulate health check (ping, status query, etc.)
            health_score = 1.0
            
            # Check response time
            if backend.response_time_ms > 5000:  # 5 seconds
                health_score -= 0.3
            
            # Check success rate
            if backend.success_rate < 0.8:
                health_score -= 0.3
            
            # Check load
            if backend.current_load > backend.max_concurrent_jobs * 0.9:
                health_score -= 0.2
            
            # Update status based on health score
            if health_score >= 0.8:
                backend.status = BackendStatus.HEALTHY
            elif health_score >= 0.6:
                backend.status = BackendStatus.DEGRADED
            elif health_score >= 0.3:
                backend.status = BackendStatus.OVERLOADED
            else:
                backend.status = BackendStatus.FAILED
            
            logger.debug(f"Backend {backend.id} health: {health_score:.2f} ({backend.status.value})")
            
        except Exception as e:
            logger.warning(f"Health check failed for backend {backend.id}: {e}")
            backend.status = BackendStatus.FAILED
    
    def _optimization_loop(self) -> None:
        """Background optimization of load balancing parameters."""
        
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                # Optimize backend weights based on performance
                self._optimize_backend_weights()
                
                # Rebalance if needed
                self._rebalance_if_needed()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    def _optimize_backend_weights(self) -> None:
        """Optimize backend weights based on historical performance."""
        
        for backend_id, backend in self.backends.items():
            history = list(self.performance_history[backend_id])
            
            if len(history) >= 5:  # Need minimum data
                # Calculate performance metrics
                response_times = [h.get('response_time', 1.0) for h in history[-10:]]
                success_rates = [h.get('success', True) for h in history[-10:]]
                
                avg_response_time = np.mean(response_times)
                avg_success_rate = np.mean(success_rates)
                
                # Calculate new weight
                # Lower response time and higher success rate = higher weight
                time_factor = max(0.1, 2.0 - avg_response_time)
                success_factor = avg_success_rate
                
                new_weight = time_factor * success_factor
                
                # Smooth weight updates
                backend.weight = 0.8 * backend.weight + 0.2 * new_weight
                
                logger.debug(f"Updated weight for {backend_id}: {backend.weight:.2f}")
    
    def _rebalance_if_needed(self) -> None:
        """Rebalance load if backends are significantly imbalanced."""
        
        if len(self.backends) < 2:
            return
        
        # Calculate load distribution
        loads = [backend.current_load for backend in self.backends.values()]
        if not loads:
            return
        
        load_std = np.std(loads)
        load_mean = np.mean(loads)
        
        # If standard deviation is high compared to mean, rebalancing may help
        if load_std > load_mean * 0.5:
            logger.info(f"High load imbalance detected (std: {load_std:.2f}, mean: {load_mean:.2f})")
            # In a full implementation, you might migrate some requests here
    
    def _auto_scaling_loop(self) -> None:
        """Background auto-scaling based on system load."""
        
        while True:
            try:
                time.sleep(self.scaling_config['cooldown_period'] / 10)  # Check frequently
                
                # Calculate overall system load
                total_capacity = sum(backend.max_concurrent_jobs for backend in self.backends.values())
                total_load = sum(backend.current_load for backend in self.backends.values())
                
                if total_capacity > 0:
                    load_ratio = total_load / total_capacity
                    
                    # Scale up if needed
                    if (load_ratio > self.scaling_config['scale_up_threshold'] and 
                        len(self.backends) < self.scaling_config['max_backends']):
                        self._scale_up()
                    
                    # Scale down if possible
                    elif (load_ratio < self.scaling_config['scale_down_threshold'] and 
                          len(self.backends) > self.scaling_config['min_backends']):
                        self._scale_down()
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(60)
    
    def _scale_up(self) -> None:
        """Scale up by adding more backend capacity."""
        logger.info("Auto-scaling: Adding backend capacity")
        
        # In a real implementation, this would:
        # 1. Provision new quantum simulator instances
        # 2. Register them with the load balancer
        # 3. Update capacity monitoring
        
        # For now, just log the action
        current_backends = len(self.backends)
        logger.info(f"Would scale up from {current_backends} backends")
    
    def _scale_down(self) -> None:
        """Scale down by removing excess backend capacity."""
        logger.info("Auto-scaling: Removing excess backend capacity")
        
        # Find the least utilized backend
        if len(self.backends) > self.scaling_config['min_backends']:
            least_utilized = min(
                self.backends.items(),
                key=lambda x: x[1].current_load
            )
            
            backend_id, backend = least_utilized
            
            # Only scale down if backend is truly idle
            if backend.current_load == 0:
                logger.info(f"Would scale down backend {backend_id}")
                # In real implementation: safely drain and remove backend
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics."""
        
        total_capacity = sum(backend.max_concurrent_jobs for backend in self.backends.values())
        total_load = sum(backend.current_load for backend in self.backends.values())
        
        backend_stats = {}
        for backend_id, backend in self.backends.items():
            backend_stats[backend_id] = {
                'name': backend.name,
                'type': backend.backend_type,
                'status': backend.status.value,
                'current_load': backend.current_load,
                'max_capacity': backend.max_concurrent_jobs,
                'utilization': backend.current_load / max(1, backend.max_concurrent_jobs),
                'response_time_ms': backend.response_time_ms,
                'success_rate': backend.success_rate,
                'weight': backend.weight,
                'queue_length': backend.queue_length
            }
        
        return {
            'strategy': self.strategy.value,
            'total_backends': len(self.backends),
            'healthy_backends': sum(1 for b in self.backends.values() if b.status == BackendStatus.HEALTHY),
            'total_capacity': total_capacity,
            'total_load': total_load,
            'overall_utilization': total_load / max(1, total_capacity),
            'active_requests': len(self.active_requests),
            'pending_requests': self.pending_requests.qsize(),
            'backend_details': backend_stats
        }