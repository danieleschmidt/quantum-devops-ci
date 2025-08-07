"""
Concurrent processing and resource pooling for quantum DevOps CI/CD.

This module provides thread pools, process pools, and resource management
for efficient parallel execution of quantum workloads.
"""

import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import queue
import logging
from contextlib import contextmanager

from .exceptions import ResourceExhaustionError, TestExecutionError
from .caching import CacheManager


@dataclass
class ResourcePool:
    """Resource pool for managing shared resources."""
    name: str
    max_size: int
    current_size: int = 0
    available_resources: List[Any] = field(default_factory=list)
    in_use_resources: Dict[str, Any] = field(default_factory=dict)
    creation_func: Optional[Callable] = None
    cleanup_func: Optional[Callable] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool."""
        start_time = time.time()
        
        while True:
            with self.lock:
                # Try to get available resource
                if self.available_resources:
                    resource = self.available_resources.pop()
                    resource_id = str(id(resource))
                    self.in_use_resources[resource_id] = resource
                    return resource
                
                # Try to create new resource
                if self.current_size < self.max_size and self.creation_func:
                    try:
                        resource = self.creation_func()
                        resource_id = str(id(resource))
                        self.in_use_resources[resource_id] = resource
                        self.current_size += 1
                        return resource
                    except Exception as e:
                        logging.error(f"Failed to create resource for pool {self.name}: {e}")
                        raise ResourceExhaustionError(f"Cannot create resource: {e}")
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise ResourceExhaustionError(f"Timeout acquiring resource from pool {self.name}")
            
            # Wait and retry
            time.sleep(0.1)
    
    def release(self, resource: Any):
        """Release a resource back to the pool."""
        with self.lock:
            resource_id = str(id(resource))
            if resource_id in self.in_use_resources:
                del self.in_use_resources[resource_id]
                self.available_resources.append(resource)
    
    def cleanup(self):
        """Clean up all resources in the pool."""
        with self.lock:
            all_resources = list(self.available_resources) + list(self.in_use_resources.values())
            
            if self.cleanup_func:
                for resource in all_resources:
                    try:
                        self.cleanup_func(resource)
                    except Exception as e:
                        logging.warning(f"Error cleaning up resource: {e}")
            
            self.available_resources.clear()
            self.in_use_resources.clear()
            self.current_size = 0


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    task_id: str
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    worker_id: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None


class ConcurrentExecutor:
    """High-level concurrent executor for quantum tasks."""
    
    def __init__(self,
                 max_threads: int = None,
                 max_processes: int = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        Initialize concurrent executor.
        
        Args:
            max_threads: Maximum number of threads
            max_processes: Maximum number of processes
            cache_manager: Optional cache manager for results
        """
        self.max_threads = max_threads or min(32, multiprocessing.cpu_count() * 4)
        self.max_processes = max_processes or multiprocessing.cpu_count()
        self.cache_manager = cache_manager
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Task tracking
        self.active_tasks: Dict[str, TaskMetrics] = {}
        self.completed_tasks: List[TaskMetrics] = []
        self.task_counter = 0
        self.lock = threading.Lock()
        
        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}
    
    def submit_thread_task(self, func: Callable, *args, **kwargs) -> Future:
        """Submit task for thread execution."""
        task_id = self._generate_task_id()
        
        with self.lock:
            metrics = TaskMetrics(
                task_id=task_id,
                submitted_at=datetime.now()
            )
            self.active_tasks[task_id] = metrics
        
        # Wrap function to track metrics
        def tracked_func():
            with self.lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].started_at = datetime.now()
                    self.active_tasks[task_id].worker_id = threading.current_thread().name
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                with self.lock:
                    if task_id in self.active_tasks:
                        metrics = self.active_tasks[task_id]
                        metrics.completed_at = datetime.now()
                        metrics.execution_time_seconds = time.time() - start_time
                        metrics.success = True
                        
                        self.completed_tasks.append(metrics)
                        del self.active_tasks[task_id]
                
                return result
                
            except Exception as e:
                with self.lock:
                    if task_id in self.active_tasks:
                        metrics = self.active_tasks[task_id]
                        metrics.completed_at = datetime.now()
                        metrics.execution_time_seconds = time.time() - start_time
                        metrics.success = False
                        metrics.error_message = str(e)
                        
                        self.completed_tasks.append(metrics)
                        del self.active_tasks[task_id]
                
                raise
        
        return self.thread_pool.submit(tracked_func)
    
    def submit_process_task(self, func: Callable, *args, **kwargs) -> Future:
        """Submit task for process execution."""
        task_id = self._generate_task_id()
        
        with self.lock:
            metrics = TaskMetrics(
                task_id=task_id,
                submitted_at=datetime.now()
            )
            self.active_tasks[task_id] = metrics
        
        future = self.process_pool.submit(func, *args, **kwargs)
        
        # Add callback to track completion
        def completion_callback(fut):
            with self.lock:
                if task_id in self.active_tasks:
                    metrics = self.active_tasks[task_id]
                    metrics.completed_at = datetime.now()
                    
                    if metrics.started_at:
                        metrics.execution_time_seconds = (
                            metrics.completed_at - metrics.started_at
                        ).total_seconds()
                    
                    if fut.exception():
                        metrics.success = False
                        metrics.error_message = str(fut.exception())
                    else:
                        metrics.success = True
                    
                    self.completed_tasks.append(metrics)
                    del self.active_tasks[task_id]
        
        future.add_done_callback(completion_callback)
        return future
    
    def map_concurrent(self, func: Callable, iterable, 
                      use_processes: bool = False, max_workers: Optional[int] = None) -> List[Any]:
        """Execute function concurrently over iterable."""
        executor = self.process_pool if use_processes else self.thread_pool
        
        if use_processes:
            # For processes, we need to be more careful about pickling
            futures = [self.submit_process_task(func, item) for item in iterable]
        else:
            futures = [self.submit_thread_task(func, item) for item in iterable]
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Task failed in concurrent map: {e}")
                results.append(None)  # Or raise depending on requirements
        
        return results
    
    def create_resource_pool(self, name: str, max_size: int,
                           creation_func: Callable,
                           cleanup_func: Optional[Callable] = None) -> ResourcePool:
        """Create a new resource pool."""
        pool = ResourcePool(
            name=name,
            max_size=max_size,
            creation_func=creation_func,
            cleanup_func=cleanup_func
        )
        
        self.resource_pools[name] = pool
        return pool
    
    @contextmanager
    def acquire_resource(self, pool_name: str, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing resources."""
        if pool_name not in self.resource_pools:
            raise ValueError(f"Resource pool '{pool_name}' not found")
        
        pool = self.resource_pools[pool_name]
        resource = pool.acquire(timeout)
        
        try:
            yield resource
        finally:
            pool.release(resource)
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        with self.lock:
            total_completed = len(self.completed_tasks)
            successful = sum(1 for task in self.completed_tasks if task.success)
            failed = total_completed - successful
            
            avg_execution_time = 0
            if self.completed_tasks:
                valid_times = [t.execution_time_seconds for t in self.completed_tasks 
                              if t.execution_time_seconds is not None]
                if valid_times:
                    avg_execution_time = sum(valid_times) / len(valid_times)
            
            return {
                'active_tasks': len(self.active_tasks),
                'completed_tasks': total_completed,
                'successful_tasks': successful,
                'failed_tasks': failed,
                'success_rate': successful / total_completed if total_completed > 0 else 0,
                'average_execution_time_seconds': avg_execution_time,
                'thread_pool_active': self.thread_pool._threads,
                'process_pool_active': len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
            }
    
    def get_resource_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get resource pool statistics."""
        stats = {}
        for name, pool in self.resource_pools.items():
            with pool.lock:
                stats[name] = {
                    'max_size': pool.max_size,
                    'current_size': pool.current_size,
                    'available': len(pool.available_resources),
                    'in_use': len(pool.in_use_resources)
                }
        return stats
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor and clean up resources."""
        logging.info("Shutting down concurrent executor...")
        
        # Shutdown thread and process pools
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)
        
        # Clean up resource pools
        for pool in self.resource_pools.values():
            pool.cleanup()
        
        logging.info("Concurrent executor shutdown complete")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self.lock:
            self.task_counter += 1
            return f"task_{self.task_counter}_{int(time.time())}"


class WorkQueue:
    """Thread-safe work queue for task distribution."""
    
    def __init__(self, maxsize: int = 0):
        """Initialize work queue."""
        self.queue = queue.Queue(maxsize)
        self.completed_tasks = queue.Queue()
        self.task_count = 0
        self.completed_count = 0
        self.lock = threading.Lock()
    
    def put_task(self, task: Any, priority: int = 0) -> str:
        """Put task in queue with optional priority."""
        with self.lock:
            task_id = f"task_{self.task_count}"
            self.task_count += 1
        
        task_wrapper = {
            'id': task_id,
            'task': task,
            'priority': priority,
            'submitted_at': datetime.now()
        }
        
        self.queue.put(task_wrapper)
        return task_id
    
    def get_task(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get task from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def mark_task_done(self, task_id: str, result: Any = None, error: Optional[str] = None):
        """Mark task as completed."""
        completion_info = {
            'task_id': task_id,
            'completed_at': datetime.now(),
            'result': result,
            'error': error
        }
        
        self.completed_tasks.put(completion_info)
        
        with self.lock:
            self.completed_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status."""
        with self.lock:
            return {
                'pending_tasks': self.queue.qsize(),
                'completed_tasks': self.completed_count,
                'total_submitted': self.task_count
            }


class Worker:
    """Worker thread for processing tasks from queue."""
    
    def __init__(self, worker_id: str, work_queue: WorkQueue, 
                 task_processor: Callable):
        """Initialize worker."""
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.task_processor = task_processor
        self.is_running = False
        self.thread = None
        self.processed_count = 0
        self.error_count = 0
    
    def start(self):
        """Start worker thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._work_loop, daemon=True)
        self.thread.start()
        logging.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop worker thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logging.info(f"Worker {self.worker_id} stopped")
    
    def _work_loop(self):
        """Main work loop."""
        while self.is_running:
            try:
                task_wrapper = self.work_queue.get_task(timeout=1.0)
                if task_wrapper is None:
                    continue
                
                task_id = task_wrapper['id']
                task = task_wrapper['task']
                
                # Process task
                try:
                    result = self.task_processor(task)
                    self.work_queue.mark_task_done(task_id, result=result)
                    self.processed_count += 1
                    
                except Exception as e:
                    logging.error(f"Worker {self.worker_id} task {task_id} failed: {e}")
                    self.work_queue.mark_task_done(task_id, error=str(e))
                    self.error_count += 1
                
            except Exception as e:
                logging.error(f"Worker {self.worker_id} encountered error: {e}")
                time.sleep(1)  # Prevent tight error loops
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'error_count': self.error_count
        }


class WorkerPool:
    """Pool of worker threads for task processing."""
    
    def __init__(self, num_workers: int, task_processor: Callable):
        """Initialize worker pool."""
        self.num_workers = num_workers
        self.task_processor = task_processor
        self.work_queue = WorkQueue()
        self.workers: List[Worker] = []
        
        # Create workers
        for i in range(num_workers):
            worker = Worker(f"worker_{i}", self.work_queue, task_processor)
            self.workers.append(worker)
    
    def start(self):
        """Start all workers."""
        for worker in self.workers:
            worker.start()
        logging.info(f"Worker pool with {self.num_workers} workers started")
    
    def stop(self):
        """Stop all workers."""
        for worker in self.workers:
            worker.stop()
        logging.info("Worker pool stopped")
    
    def submit_task(self, task: Any, priority: int = 0) -> str:
        """Submit task to worker pool."""
        return self.work_queue.put_task(task, priority)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        worker_stats = [worker.get_stats() for worker in self.workers]
        queue_stats = self.work_queue.get_status()
        
        total_processed = sum(w['processed_count'] for w in worker_stats)
        total_errors = sum(w['error_count'] for w in worker_stats)
        
        return {
            'num_workers': self.num_workers,
            'queue_status': queue_stats,
            'total_processed': total_processed,
            'total_errors': total_errors,
            'worker_details': worker_stats
        }


# Utility functions for quantum-specific concurrent processing

def execute_circuits_concurrently(circuits: List[Any], 
                                 executor_func: Callable,
                                 max_workers: int = None,
                                 use_processes: bool = False) -> List[Any]:
    """Execute multiple quantum circuits concurrently."""
    if not circuits:
        return []
    
    executor = ConcurrentExecutor(
        max_threads=max_workers,
        max_processes=max_workers
    )
    
    try:
        if use_processes:
            results = executor.map_concurrent(executor_func, circuits, use_processes=True)
        else:
            results = executor.map_concurrent(executor_func, circuits, use_processes=False)
        
        return results
    finally:
        executor.shutdown()


def batch_process_with_cache(items: List[Any],
                           processor_func: Callable,
                           cache_manager: CacheManager,
                           cache_key_func: Callable,
                           batch_size: int = 10,
                           max_workers: int = None) -> List[Any]:
    """Process items in batches with caching support."""
    results = []
    uncached_items = []
    uncached_indices = []
    
    # Check cache for all items first
    for i, item in enumerate(items):
        cache_key = cache_key_func(item)
        cached_result = cache_manager.circuit_cache.get(cache_key)
        
        if cached_result is not None:
            results.append(cached_result)
        else:
            results.append(None)  # Placeholder
            uncached_items.append(item)
            uncached_indices.append(i)
    
    if not uncached_items:
        return results
    
    # Process uncached items concurrently
    executor = ConcurrentExecutor(max_threads=max_workers)
    
    try:
        # Split into batches
        batches = [uncached_items[i:i + batch_size] 
                  for i in range(0, len(uncached_items), batch_size)]
        
        def process_batch(batch):
            batch_results = []
            for item in batch:
                result = processor_func(item)
                
                # Cache the result
                cache_key = cache_key_func(item)
                cache_manager.circuit_cache.put(cache_key, result)
                
                batch_results.append(result)
            return batch_results
        
        # Process all batches concurrently
        batch_futures = [executor.submit_thread_task(process_batch, batch) 
                        for batch in batches]
        
        # Collect results
        batch_results = []
        for future in batch_futures:
            batch_results.extend(future.result())
        
        # Fill in results at correct indices
        for i, result in zip(uncached_indices, batch_results):
            results[i] = result
        
        return results
        
    finally:
        executor.shutdown()