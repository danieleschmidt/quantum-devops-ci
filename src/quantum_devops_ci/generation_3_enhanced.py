"""
Generation 3 Enhanced Features - Advanced performance optimization and scaling systems.

This module provides enhanced Generation 3 functionality including:
- Adaptive load balancing and distribution
- Advanced caching strategies with ML-based optimization
- Auto-scaling with predictive analytics
- Resource pooling and connection management
- Performance monitoring and optimization
- Distributed computing capabilities
"""

import logging
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import hashlib
import queue
import concurrent.futures
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategy types."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    ADAPTIVE_ML = "adaptive_ml"
    RESOURCE_BASED = "resource_based"

class ScalingStrategy(Enum):
    """Auto-scaling strategy types."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"

@dataclass
class WorkerNode:
    """Worker node information."""
    node_id: str
    host: str
    port: int
    current_load: float = 0.0
    max_capacity: int = 100
    current_connections: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    health_score: float = 1.0
    last_heartbeat: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

@dataclass
class CacheEntry:
    """Enhanced cache entry with ML optimization."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    size_bytes: int = 0
    frequency_score: float = 0.0
    prediction_score: float = 0.0
    
class AdvancedLoadBalancer:
    """Advanced load balancer with ML-based optimization."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_ML):
        """Initialize load balancer."""
        self.strategy = strategy
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.request_history: deque = deque(maxlen=10000)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.ml_model_weights: Dict[str, float] = {
            'response_time': 0.4,
            'success_rate': 0.3,
            'current_load': 0.2,
            'health_score': 0.1
        }
        self.lock = threading.RLock()
        
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node."""
        with self.lock:
            self.worker_nodes[worker.node_id] = worker
            logger.info(f"Registered worker node: {worker.node_id}")
    
    def select_worker(self, request_context: Optional[Dict] = None) -> Optional[WorkerNode]:
        """Select optimal worker based on strategy."""
        with self.lock:
            if not self.worker_nodes:
                return None
            
            available_workers = [
                worker for worker in self.worker_nodes.values()
                if worker.health_score > 0.5 and worker.current_connections < worker.max_capacity
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(available_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(available_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
                return self._select_weighted_response_time(available_workers)
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE_ML:
                return self._select_adaptive_ml(available_workers, request_context)
            else:
                return self._select_resource_based(available_workers)
    
    def _select_round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin selection."""
        # Simple round-robin based on total requests
        return min(workers, key=lambda w: w.total_requests)
    
    def _select_least_connections(self, workers: List[WorkerNode]) -> WorkerNode:
        """Least connections selection."""
        return min(workers, key=lambda w: w.current_connections)
    
    def _select_weighted_response_time(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted response time selection."""
        def weight_score(worker):
            response_time = worker.average_response_time or 1.0
            return response_time * (1 + worker.current_load)
        
        return min(workers, key=weight_score)
    
    def _select_adaptive_ml(self, workers: List[WorkerNode], context: Optional[Dict]) -> WorkerNode:
        """ML-based adaptive selection."""
        def ml_score(worker):
            score = 0.0
            
            # Response time factor (lower is better)
            response_factor = 1.0 / (worker.average_response_time + 0.1)
            score += self.ml_model_weights['response_time'] * response_factor
            
            # Success rate factor
            score += self.ml_model_weights['success_rate'] * worker.success_rate
            
            # Load factor (lower load is better)
            load_factor = 1.0 - worker.current_load
            score += self.ml_model_weights['current_load'] * load_factor
            
            # Health score factor
            score += self.ml_model_weights['health_score'] * worker.health_score
            
            return score
        
        return max(workers, key=ml_score)
    
    def _select_resource_based(self, workers: List[WorkerNode]) -> WorkerNode:
        """Resource-based selection."""
        def resource_score(worker):
            # Combined score based on capacity and current usage
            utilization = worker.current_connections / worker.max_capacity
            return (1.0 - utilization) * worker.health_score
        
        return max(workers, key=resource_score)
    
    def update_worker_metrics(self, worker_id: str, response_time: float, success: bool):
        """Update worker performance metrics."""
        with self.lock:
            if worker_id in self.worker_nodes:
                worker = self.worker_nodes[worker_id]
                worker.response_times.append(response_time)
                worker.total_requests += 1
                if success:
                    worker.successful_requests += 1
                
                # Update ML model weights based on performance
                self._update_ml_weights(worker, response_time, success)
    
    def _update_ml_weights(self, worker: WorkerNode, response_time: float, success: bool):
        """Update ML model weights based on performance feedback."""
        # Simple adaptive weight adjustment
        if success and response_time < 1.0:  # Good performance
            self.ml_model_weights['response_time'] = min(0.5, self.ml_model_weights['response_time'] * 1.01)
        elif not success or response_time > 5.0:  # Poor performance
            self.ml_model_weights['response_time'] = max(0.1, self.ml_model_weights['response_time'] * 0.99)
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            total_requests = sum(w.total_requests for w in self.worker_nodes.values())
            total_successful = sum(w.successful_requests for w in self.worker_nodes.values())
            
            return {
                'strategy': self.strategy.value,
                'total_workers': len(self.worker_nodes),
                'healthy_workers': len([w for w in self.worker_nodes.values() if w.health_score > 0.5]),
                'total_requests': total_requests,
                'success_rate': total_successful / max(1, total_requests),
                'ml_weights': self.ml_model_weights.copy(),
                'worker_stats': {
                    worker_id: {
                        'requests': worker.total_requests,
                        'success_rate': worker.success_rate,
                        'avg_response_time': worker.average_response_time,
                        'current_load': worker.current_load
                    }
                    for worker_id, worker in self.worker_nodes.items()
                }
            }

class IntelligentCache:
    """Intelligent cache with ML-based optimization."""
    
    def __init__(self, max_size: int = 10000, eviction_strategy: str = "ml_optimized"):
        """Initialize intelligent cache."""
        self.max_size = max_size
        self.eviction_strategy = eviction_strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.size_tracker = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        # ML optimization parameters
        self.frequency_weight = 0.4
        self.recency_weight = 0.3
        self.size_weight = 0.2
        self.prediction_weight = 0.1
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = datetime.now()
                self.access_patterns[key].append(datetime.now())
                self.hits += 1
                
                # Update frequency score
                self._update_frequency_score(entry)
                
                return entry.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, size_hint: Optional[int] = None) -> bool:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing entry
                old_entry = self.cache[key]
                self.size_tracker -= old_entry.size_bytes
            
            # Calculate size
            value_size = size_hint or self._calculate_size(value)
            
            # Check if we need to evict entries
            while (self.size_tracker + value_size > self.max_size * 1000 and  # Assume max_size in KB
                   len(self.cache) > 0):
                self._evict_entry()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                access_count=1,
                last_access=datetime.now(),
                size_bytes=value_size
            )
            
            self.cache[key] = entry
            self.size_tracker += value_size
            self.access_patterns[key].append(datetime.now())
            
            # Update prediction score
            self._update_prediction_score(entry)
            
            return True
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value."""
        try:
            return len(str(value).encode('utf-8'))
        except:
            return 1000  # Default size
    
    def _update_frequency_score(self, entry: CacheEntry):
        """Update frequency score based on access patterns."""
        now = datetime.now()
        recent_accesses = [
            access for access in self.access_patterns.get(entry.key, [])
            if (now - access).total_seconds() < 3600  # Last hour
        ]
        entry.frequency_score = len(recent_accesses) / 60.0  # Accesses per minute
    
    def _update_prediction_score(self, entry: CacheEntry):
        """Update prediction score using simple pattern analysis."""
        access_times = self.access_patterns.get(entry.key, [])
        if len(access_times) >= 2:
            # Simple pattern: regular intervals suggest future access
            intervals = [
                (access_times[i] - access_times[i-1]).total_seconds()
                for i in range(1, len(access_times))
            ]
            if intervals:
                avg_interval = statistics.mean(intervals)
                # Predict based on regularity
                entry.prediction_score = 1.0 / (1.0 + abs(avg_interval - 300))  # 5-min optimal
    
    def _evict_entry(self):
        """Evict entry based on strategy."""
        if not self.cache:
            return
        
        if self.eviction_strategy == "ml_optimized":
            self._evict_ml_optimized()
        elif self.eviction_strategy == "lru":
            self._evict_lru()
        else:
            self._evict_lfu()
    
    def _evict_ml_optimized(self):
        """ML-optimized eviction."""
        def eviction_score(entry):
            # Lower score = more likely to evict
            now = datetime.now()
            
            # Recency factor
            time_since_access = (now - entry.last_access).total_seconds() if entry.last_access else float('inf')
            recency_factor = 1.0 / (1.0 + time_since_access / 3600)  # Normalize to hours
            
            # Frequency factor
            frequency_factor = entry.frequency_score
            
            # Size factor (larger entries more likely to be evicted)
            size_factor = 1.0 / (1.0 + entry.size_bytes / 1000)  # Normalize to KB
            
            # Prediction factor
            prediction_factor = entry.prediction_score
            
            return (self.frequency_weight * frequency_factor +
                   self.recency_weight * recency_factor +
                   self.size_weight * size_factor +
                   self.prediction_weight * prediction_factor)
        
        # Evict entry with lowest score
        entry_to_evict = min(self.cache.values(), key=eviction_score)
        self._remove_entry(entry_to_evict.key)
    
    def _evict_lru(self):
        """Least Recently Used eviction."""
        lru_entry = min(self.cache.values(), key=lambda e: e.last_access or datetime.min)
        self._remove_entry(lru_entry.key)
    
    def _evict_lfu(self):
        """Least Frequently Used eviction."""
        lfu_entry = min(self.cache.values(), key=lambda e: e.access_count)
        self._remove_entry(lfu_entry.key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.size_tracker -= entry.size_bytes
            del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'size_bytes': self.size_tracker,
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses,
                'eviction_strategy': self.eviction_strategy,
                'ml_weights': {
                    'frequency': self.frequency_weight,
                    'recency': self.recency_weight,
                    'size': self.size_weight,
                    'prediction': self.prediction_weight
                }
            }

class PredictiveAutoScaler:
    """Predictive auto-scaling system."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        """Initialize auto-scaler."""
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_events: List[Dict] = []
        self.prediction_model = SimplePredictor()
        
    def record_metrics(self, cpu_usage: float, memory_usage: float, request_rate: float):
        """Record system metrics."""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'request_rate': request_rate,
            'instances': self.current_instances
        }
        self.metrics_history.append(metrics)
        
        # Make scaling decision
        self._evaluate_scaling()
    
    def _evaluate_scaling(self):
        """Evaluate if scaling is needed."""
        if len(self.metrics_history) < 5:
            return
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-5:]
        avg_cpu = statistics.mean(m['cpu_usage'] for m in recent_metrics)
        avg_memory = statistics.mean(m['memory_usage'] for m in recent_metrics)
        avg_request_rate = statistics.mean(m['request_rate'] for m in recent_metrics)
        
        # Predict future load
        predicted_load = self.prediction_model.predict_load(recent_metrics)
        
        # Make scaling decision
        if predicted_load > 0.8 and self.current_instances < self.max_instances:
            self._scale_up()
        elif predicted_load < 0.3 and self.current_instances > self.min_instances:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up instances."""
        old_instances = self.current_instances
        self.current_instances = min(self.max_instances, self.current_instances + 1)
        
        event = {
            'timestamp': datetime.now(),
            'action': 'scale_up',
            'from_instances': old_instances,
            'to_instances': self.current_instances,
            'reason': 'high_load_prediction'
        }
        self.scaling_events.append(event)
        logger.info(f"Scaled up: {old_instances} → {self.current_instances} instances")
    
    def _scale_down(self):
        """Scale down instances."""
        old_instances = self.current_instances
        self.current_instances = max(self.min_instances, self.current_instances - 1)
        
        event = {
            'timestamp': datetime.now(),
            'action': 'scale_down',
            'from_instances': old_instances,
            'to_instances': self.current_instances,
            'reason': 'low_load_prediction'
        }
        self.scaling_events.append(event)
        logger.info(f"Scaled down: {old_instances} → {self.current_instances} instances")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'recent_avg_cpu': statistics.mean(m['cpu_usage'] for m in recent_metrics),
            'recent_avg_memory': statistics.mean(m['memory_usage'] for m in recent_metrics),
            'recent_avg_requests': statistics.mean(m['request_rate'] for m in recent_metrics),
            'total_scaling_events': len(self.scaling_events),
            'recent_scaling_events': len([e for e in self.scaling_events 
                                        if (datetime.now() - e['timestamp']).total_seconds() < 86400])
        }

class SimplePredictor:
    """Simple load prediction model."""
    
    def predict_load(self, metrics_history: List[Dict]) -> float:
        """Predict future load based on historical metrics."""
        if len(metrics_history) < 3:
            return 0.5  # Default prediction
        
        # Simple trend-based prediction
        cpu_values = [m['cpu_usage'] for m in metrics_history]
        memory_values = [m['memory_usage'] for m in metrics_history]
        request_values = [m['request_rate'] for m in metrics_history]
        
        # Calculate trends
        cpu_trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
        memory_trend = (memory_values[-1] - memory_values[0]) / len(memory_values)
        request_trend = (request_values[-1] - request_values[0]) / len(request_values)
        
        # Combine trends into load prediction
        predicted_cpu = cpu_values[-1] + cpu_trend * 2  # Predict 2 steps ahead
        predicted_memory = memory_values[-1] + memory_trend * 2
        predicted_requests = request_values[-1] + request_trend * 2
        
        # Normalize and combine
        load_score = (predicted_cpu + predicted_memory + predicted_requests / 100) / 3
        return max(0.0, min(1.0, load_score))

class Generation3EnhancedDemo:
    """Demonstration of Generation 3 enhanced features."""
    
    def __init__(self):
        self.load_balancer = AdvancedLoadBalancer()
        self.cache = IntelligentCache()
        self.auto_scaler = PredictiveAutoScaler()
        
        # Setup demo workers
        self._setup_demo_workers()
    
    def _setup_demo_workers(self):
        """Setup demo worker nodes."""
        workers = [
            WorkerNode("worker_1", "192.168.1.10", 8080, max_capacity=50),
            WorkerNode("worker_2", "192.168.1.11", 8080, max_capacity=75),
            WorkerNode("worker_3", "192.168.1.12", 8080, max_capacity=100),
        ]
        
        for worker in workers:
            self.load_balancer.register_worker(worker)
    
    def run_performance_demo(self) -> Dict[str, Any]:
        """Run performance optimization demonstration."""
        print("⚡ Starting Performance Optimization Demo...")
        
        # Simulate load balancing
        for i in range(10):
            worker = self.load_balancer.select_worker()
            if worker:
                # Simulate request processing
                response_time = 0.5 + (i % 3) * 0.2
                success = True
                self.load_balancer.update_worker_metrics(worker.node_id, response_time, success)
        
        # Test cache performance
        for i in range(20):
            key = f"cache_key_{i % 5}"  # Some repetition for hits
            if self.cache.get(key) is None:
                self.cache.put(key, f"value_{i}", size_hint=100)
        
        # Simulate auto-scaling metrics
        for i in range(15):
            cpu_usage = 0.3 + (i / 15) * 0.6  # Gradually increase
            memory_usage = 0.2 + (i / 15) * 0.5
            request_rate = 10 + i * 5
            self.auto_scaler.record_metrics(cpu_usage, memory_usage, request_rate)
        
        return {
            'load_balancing': self.load_balancer.get_load_balancing_stats(),
            'caching': self.cache.get_cache_stats(),
            'auto_scaling': self.auto_scaler.get_scaling_stats()
        }

def run_generation_3_enhanced_demo():
    """Run complete Generation 3 enhanced demonstration."""
    print("⚡ Generation 3 Enhanced Features Demo")
    print("=" * 50)
    
    demo = Generation3EnhancedDemo()
    
    # Run performance demo
    print("\n📊 Performance Optimization Demo:")
    perf_results = demo.run_performance_demo()
    
    print(f"  ✅ Load Balancing: {perf_results['load_balancing']['total_workers']} workers, "
          f"{perf_results['load_balancing']['success_rate']:.1%} success rate")
    print(f"  ✅ Intelligent Cache: {perf_results['caching']['hit_rate']:.1%} hit rate, "
          f"{perf_results['caching']['size']} entries")
    print(f"  ✅ Auto-scaling: {perf_results['auto_scaling']['current_instances']} instances, "
          f"{perf_results['auto_scaling']['total_scaling_events']} scaling events")
    
    print("\n✨ Generation 3 Enhanced features successfully demonstrated!")
    return perf_results

if __name__ == "__main__":
    run_generation_3_enhanced_demo()