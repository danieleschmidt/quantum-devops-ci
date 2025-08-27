"""
High-Performance Quantum Computing Framework.
Generation 3: MAKE IT SCALE - Performance optimization, caching, concurrent processing.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable, AsyncIterator
from enum import Enum
import weakref

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .exceptions import ResourceExhaustionError, TestExecutionError
from .caching import CacheManager
from .monitoring import QuantumCIMonitor
from .resilience import CircuitBreakerConfig, RetryPolicy


logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Real-time performance metrics collection."""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [], 
            'execution_times': [],
            'throughput': [],
            'concurrent_tasks': [],
            'cache_hit_rate': 0.0,
            'errors_per_minute': 0.0
        }
        self._lock = threading.Lock()
        
    def record_cpu_usage(self, usage: float):
        """Record CPU usage percentage."""
        with self._lock:
            self.metrics['cpu_usage'].append((time.time(), usage))
            # Keep only last 100 measurements
            if len(self.metrics['cpu_usage']) > 100:
                self.metrics['cpu_usage'] = self.metrics['cpu_usage'][-100:]
    
    def record_memory_usage(self, usage_mb: float):
        """Record memory usage in MB."""
        with self._lock:
            self.metrics['memory_usage'].append((time.time(), usage_mb))
            if len(self.metrics['memory_usage']) > 100:
                self.metrics['memory_usage'] = self.metrics['memory_usage'][-100:]
    
    def record_execution_time(self, time_seconds: float):
        """Record task execution time."""
        with self._lock:
            self.metrics['execution_times'].append((time.time(), time_seconds))
            if len(self.metrics['execution_times']) > 1000:
                self.metrics['execution_times'] = self.metrics['execution_times'][-1000:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance snapshot."""
        with self._lock:
            return {
                'avg_cpu_usage': self._calculate_average('cpu_usage'),
                'avg_memory_usage': self._calculate_average('memory_usage'),
                'avg_execution_time': self._calculate_average('execution_times'),
                'current_throughput': self._calculate_throughput(),
                'cache_hit_rate': self.metrics['cache_hit_rate'],
                'total_measurements': sum(len(v) for v in self.metrics.values() if isinstance(v, list))
            }
    
    def _calculate_average(self, metric_name: str) -> float:
        """Calculate average for time-series metric."""
        data = self.metrics.get(metric_name, [])
        if not data:
            return 0.0
        return sum(value for _, value in data) / len(data)
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks per second."""
        now = time.time()
        recent_tasks = [t for t, _ in self.metrics['execution_times'] if now - t < 60]
        return len(recent_tasks) / 60.0 if recent_tasks else 0.0


@dataclass
class ScalingConfig:
    """Configuration for high-performance scaling."""
    max_workers: int = min(32, (multiprocessing.cpu_count() or 1) + 4)
    max_memory_usage_percent: float = 85.0
    max_cpu_usage_percent: float = 90.0
    auto_scaling_enabled: bool = True
    performance_monitoring_interval: float = 5.0
    cache_size_mb: int = 512
    batch_size: int = 100
    connection_pool_size: int = 20
    prefetch_buffer_size: int = 1000
    enable_jit_compilation: bool = True
    enable_vectorization: bool = True
    enable_gpu_acceleration: bool = False


class ResourcePool:
    """Optimized resource pool with auto-scaling capabilities."""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.metrics = PerformanceMetrics()
        
        # Thread and process pools
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(self.config.max_workers, multiprocessing.cpu_count())
        )
        
        # Connection pools and caches
        self.connection_pool = self._create_connection_pool()
        self.cache_manager = CacheManager()
        
        # Performance monitoring
        self._monitoring_task = None
        self._shutdown_event = asyncio.Event()
        
        # Resource tracking
        self.active_tasks = weakref.WeakSet()
        self._resource_lock = asyncio.Lock()
        
        logger.info(f"ResourcePool initialized with {self.config.max_workers} max workers")
    
    def _create_connection_pool(self) -> Dict[str, Any]:
        """Create optimized connection pool."""
        return {
            'active_connections': [],
            'idle_connections': [],
            'max_connections': self.config.connection_pool_size,
            'current_connections': 0
        }
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_performance())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self._monitoring_task:
            self._shutdown_event.set()
            await self._monitoring_task
            self._monitoring_task = None
            logger.info("Performance monitoring stopped")
    
    async def _monitor_performance(self):
        """Continuous performance monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._collect_metrics()
                await self._optimize_resources()
                await asyncio.sleep(self.config.performance_monitoring_interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _collect_metrics(self):
        """Collect current system metrics."""
        if PSUTIL_AVAILABLE:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.record_cpu_usage(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.record_memory_usage(memory.used / 1024 / 1024)
            
            # Check for resource pressure
            if cpu_percent > self.config.max_cpu_usage_percent:
                logger.warning(f"High CPU usage detected: {cpu_percent}%")
                
            if memory.percent > self.config.max_memory_usage_percent:
                logger.warning(f"High memory usage detected: {memory.percent}%")
    
    async def _optimize_resources(self):
        """Dynamic resource optimization."""
        metrics = self.metrics.get_current_metrics()
        
        # Auto-scaling decision
        if self.config.auto_scaling_enabled:
            await self._auto_scale_resources(metrics)
        
        # Cache optimization
        await self._optimize_cache()
        
        # Connection pool optimization
        await self._optimize_connections()
    
    async def _auto_scale_resources(self, metrics: Dict[str, Any]):
        """Automatically scale resources based on metrics."""
        current_throughput = metrics['current_throughput']
        avg_cpu = metrics['avg_cpu_usage']
        avg_memory = metrics['avg_memory_usage']
        
        # Scale up conditions
        if (avg_cpu > 70 and current_throughput > 10) or len(self.active_tasks) > self.config.max_workers * 0.8:
            await self._scale_up()
        
        # Scale down conditions  
        elif avg_cpu < 30 and current_throughput < 2 and len(self.active_tasks) < self.config.max_workers * 0.2:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up resources."""
        new_max = min(self.config.max_workers + 4, 64)
        if new_max > self.config.max_workers:
            logger.info(f"Scaling up workers from {self.config.max_workers} to {new_max}")
            self.config.max_workers = new_max
    
    async def _scale_down(self):
        """Scale down resources."""
        new_max = max(self.config.max_workers - 2, 4)
        if new_max < self.config.max_workers:
            logger.info(f"Scaling down workers from {self.config.max_workers} to {new_max}")
            self.config.max_workers = new_max
    
    async def _optimize_cache(self):
        """Optimize cache performance."""
        if self.cache_manager:
            await self.cache_manager.optimize()
    
    async def _optimize_connections(self):
        """Optimize connection pool."""
        pool = self.connection_pool
        
        # Clean up idle connections
        current_time = time.time()
        pool['idle_connections'] = [
            conn for conn in pool['idle_connections'] 
            if current_time - conn.get('last_used', 0) < 300  # 5 minutes
        ]
    
    @asynccontextmanager
    async def acquire_worker(self):
        """Acquire a worker from the pool."""
        start_time = time.time()
        task_ref = weakref.ref(asyncio.current_task())
        
        try:
            async with self._resource_lock:
                if len(self.active_tasks) >= self.config.max_workers:
                    raise ResourceExhaustionError("No workers available")
                
                self.active_tasks.add(task_ref)
            
            yield self.thread_pool
            
        finally:
            execution_time = time.time() - start_time
            self.metrics.record_execution_time(execution_time)
            
            async with self._resource_lock:
                self.active_tasks.discard(task_ref)
    
    def shutdown(self):
        """Shutdown resource pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("ResourcePool shutdown completed")


class HighPerformanceQuantumExecutor:
    """High-performance quantum task executor with optimization."""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.resource_pool = ResourcePool(config)
        self.task_queue = asyncio.Queue(maxsize=self.config.prefetch_buffer_size)
        self.result_cache = {}
        self.execution_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'errors': 0
        }
        
    async def start(self):
        """Start the high-performance executor."""
        await self.resource_pool.start_monitoring()
        logger.info("HighPerformanceQuantumExecutor started")
    
    async def stop(self):
        """Stop the executor and cleanup resources."""
        await self.resource_pool.stop_monitoring()
        self.resource_pool.shutdown()
        logger.info("HighPerformanceQuantumExecutor stopped")
    
    async def execute_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of quantum tasks with optimization."""
        if not tasks:
            return []
        
        # Optimize batch size
        batch_size = min(len(tasks), self.config.batch_size)
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        results = []
        for batch in batches:
            batch_results = await self._execute_batch_optimized(batch)
            results.extend(batch_results)
        
        return results
    
    async def _execute_batch_optimized(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute optimized batch with caching and parallelization."""
        # Check cache first
        cached_results = []
        uncached_tasks = []
        
        for task in batch:
            cache_key = self._generate_cache_key(task)
            if cache_key in self.result_cache:
                cached_results.append(self.result_cache[cache_key])
                self.execution_stats['cache_hits'] += 1
            else:
                uncached_tasks.append(task)
        
        # Execute uncached tasks in parallel
        if uncached_tasks:
            parallel_results = await self._execute_parallel(uncached_tasks)
            
            # Cache results
            for task, result in zip(uncached_tasks, parallel_results):
                cache_key = self._generate_cache_key(task)
                self.result_cache[cache_key] = result
            
            cached_results.extend(parallel_results)
        
        return cached_results
    
    async def _execute_parallel(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel using resource pool."""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def execute_single_task(task):
            async with semaphore:
                async with self.resource_pool.acquire_worker() as worker:
                    return await self._execute_single_task_optimized(task, worker)
        
        # Execute all tasks concurrently
        coroutines = [execute_single_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task execution failed: {result}")
                self.execution_stats['errors'] += 1
                processed_results.append({'error': str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_task_optimized(self, task: Dict[str, Any], worker) -> Dict[str, Any]:
        """Execute single quantum task with optimizations."""
        start_time = time.time()
        
        try:
            # Apply optimizations based on task type
            if task.get('type') == 'circuit_execution':
                result = await self._execute_circuit_optimized(task, worker)
            elif task.get('type') == 'noise_simulation':
                result = await self._execute_noise_simulation_optimized(task, worker)
            else:
                result = await self._execute_generic_task(task, worker)
            
            execution_time = time.time() - start_time
            self.execution_stats['total_executions'] += 1
            self.execution_stats['total_time'] += execution_time
            
            return {
                'result': result,
                'execution_time': execution_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {
                'error': str(e),
                'execution_time': time.time() - start_time,
                'status': 'failed'
            }
    
    async def _execute_circuit_optimized(self, task: Dict[str, Any], worker) -> Any:
        """Execute quantum circuit with performance optimizations."""
        circuit = task.get('circuit')
        shots = task.get('shots', 1000)
        
        # Optimization: Use vectorization if available
        if self.config.enable_vectorization and NUMPY_AVAILABLE:
            return await self._execute_vectorized(circuit, shots, worker)
        else:
            return await self._execute_standard(circuit, shots, worker)
    
    async def _execute_vectorized(self, circuit: Any, shots: int, worker) -> Any:
        """Execute with vectorization optimization."""
        # Mock vectorized execution
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(worker, self._mock_vectorized_execution, circuit, shots)
    
    def _mock_vectorized_execution(self, circuit: Any, shots: int) -> Dict[str, Any]:
        """Mock vectorized quantum execution."""
        if NUMPY_AVAILABLE:
            # Simulate optimized execution
            prob_distribution = np.random.dirichlet(np.ones(4))  # 4 possible outcomes for 2 qubits
            states = ['00', '01', '10', '11']
            counts = {state: int(prob * shots) for state, prob in zip(states, prob_distribution)}
            
            return {
                'counts': counts,
                'shots': shots,
                'optimization': 'vectorized',
                'execution_time': 0.05  # Simulated fast execution
            }
        else:
            return {'error': 'NumPy not available for vectorization'}
    
    async def _execute_standard(self, circuit: Any, shots: int, worker) -> Any:
        """Standard execution fallback."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(worker, self._mock_standard_execution, circuit, shots)
    
    def _mock_standard_execution(self, circuit: Any, shots: int) -> Dict[str, Any]:
        """Mock standard quantum execution."""
        # Simulate standard execution
        counts = {'00': shots // 2, '11': shots // 2}
        return {
            'counts': counts,
            'shots': shots,
            'optimization': 'standard',
            'execution_time': 0.1
        }
    
    async def _execute_noise_simulation_optimized(self, task: Dict[str, Any], worker) -> Any:
        """Execute noise simulation with optimizations."""
        # Placeholder for noise simulation optimization
        return await self._execute_generic_task(task, worker)
    
    async def _execute_generic_task(self, task: Dict[str, Any], worker) -> Any:
        """Execute generic quantum task."""
        # Placeholder for generic task execution
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(worker, lambda: {'result': 'completed'})
    
    def _generate_cache_key(self, task: Dict[str, Any]) -> str:
        """Generate cache key for task."""
        # Simple hash-based key generation
        import hashlib
        task_str = str(sorted(task.items()))
        return hashlib.md5(task_str.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.execution_stats.copy()
        
        if stats['total_executions'] > 0:
            stats['average_execution_time'] = stats['total_time'] / stats['total_executions']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_executions']
        else:
            stats['average_execution_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Add resource pool metrics
        stats.update(self.resource_pool.metrics.get_current_metrics())
        
        return stats


# Convenience functions for easy integration
async def create_high_performance_executor(config: ScalingConfig = None) -> HighPerformanceQuantumExecutor:
    """Create and start a high-performance quantum executor."""
    executor = HighPerformanceQuantumExecutor(config)
    await executor.start()
    return executor


async def execute_quantum_tasks_optimized(tasks: List[Dict[str, Any]], 
                                        config: ScalingConfig = None) -> List[Dict[str, Any]]:
    """Execute quantum tasks with high-performance optimization."""
    executor = await create_high_performance_executor(config)
    
    try:
        results = await executor.execute_batch(tasks)
        return results
    finally:
        await executor.stop()