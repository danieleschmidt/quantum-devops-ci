"""
Quantum job scheduling and resource management.

This module provides intelligent scheduling of quantum jobs across different
hardware providers, considering cost, time, and hardware constraints.
"""

import abc
import warnings
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq
from enum import Enum
import yaml


class Priority(Enum):
    """Job priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QuantumJob:
    """Container for quantum job information."""
    id: str
    circuit: Any  # Quantum circuit object
    shots: int
    priority: Priority = Priority.MEDIUM
    backend_requirements: Optional[List[str]] = None
    max_execution_time: Optional[int] = None  # seconds
    deadline: Optional[datetime] = None
    cost_limit: Optional[float] = None  # USD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.backend_requirements is None:
            self.backend_requirements = []


@dataclass
class BackendInfo:
    """Information about quantum backend."""
    name: str
    provider: str
    num_qubits: int
    queue_length: int
    estimated_wait_time: float  # minutes
    cost_per_shot: float  # USD
    availability: bool = True
    calibration_data: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.calibration_data is None:
            self.calibration_data = {}


@dataclass
class ScheduleEntry:
    """Single entry in execution schedule."""
    job: QuantumJob
    backend: BackendInfo
    estimated_start_time: datetime
    estimated_end_time: datetime
    estimated_cost: float
    priority_score: float


@dataclass
class OptimizedSchedule:
    """Result of schedule optimization."""
    entries: List[ScheduleEntry]
    total_cost: float
    total_time_hours: float
    device_allocation: Dict[str, int]
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_cost = sum(entry.estimated_cost for entry in self.entries)
        if self.entries:
            start_time = min(entry.estimated_start_time for entry in self.entries)
            end_time = max(entry.estimated_end_time for entry in self.entries)
            self.total_time_hours = (end_time - start_time).total_seconds() / 3600
        else:
            self.total_time_hours = 0
        
        # Calculate device allocation
        device_counts = {}
        for entry in self.entries:
            device_name = entry.backend.name
            device_counts[device_name] = device_counts.get(device_name, 0) + 1
        self.device_allocation = device_counts


class QuantumJobScheduler:
    """
    Intelligent scheduler for quantum jobs.
    
    This scheduler optimizes job execution across multiple quantum backends
    considering cost, time, hardware constraints, and job priorities.
    """
    
    def __init__(self, config_file: Optional[str] = None, optimization_goal: str = "minimize_cost"):
        """
        Initialize quantum job scheduler.
        
        Args:
            config_file: Path to scheduler configuration file
            optimization_goal: "minimize_cost" or "minimize_time"
        """
        self.config = self._load_config(config_file)
        self.optimization_goal = optimization_goal
        self.backends = {}
        self.job_queue = []
        self.completed_jobs = []
        
        # Initialize available backends
        self._initialize_backends()
    
    def add_job(self, job: QuantumJob) -> str:
        """
        Add job to scheduling queue.
        
        Args:
            job: Quantum job to schedule
            
        Returns:
            Job ID for tracking
        """
        # Add to priority queue (negative priority for max heap behavior)
        priority_score = self._calculate_priority_score(job)
        heapq.heappush(self.job_queue, (-priority_score, job.id, job))
        
        return job.id
    
    def optimize_schedule(
        self,
        jobs: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizedSchedule:
        """
        Optimize job schedule for given constraints.
        
        Args:
            jobs: List of job specifications
            constraints: Scheduling constraints (deadline, budget, etc.)
            
        Returns:
            Optimized execution schedule
        """
        if constraints is None:
            constraints = {}
        
        # Convert job specifications to QuantumJob objects
        quantum_jobs = []
        for i, job_spec in enumerate(jobs):
            job = QuantumJob(
                id=f"job_{i}",
                circuit=job_spec["circuit"],
                shots=job_spec["shots"],
                priority=Priority(job_spec.get("priority", "medium").upper()),
                deadline=constraints.get("deadline"),
                cost_limit=constraints.get("budget")
            )
            quantum_jobs.append(job)
        
        # Get available backends
        available_backends = self._get_available_backends(constraints)
        
        # Optimize schedule based on goal
        if self.optimization_goal == "minimize_cost":
            schedule = self._optimize_for_cost(quantum_jobs, available_backends, constraints)
        elif self.optimization_goal == "minimize_time":
            schedule = self._optimize_for_time(quantum_jobs, available_backends, constraints)
        else:
            raise ValueError(f"Unknown optimization goal: {self.optimization_goal}")
        
        return schedule
    
    def submit_job(self, job: QuantumJob, backend_name: Optional[str] = None) -> str:
        """
        Submit job for execution.
        
        Args:
            job: Quantum job to submit
            backend_name: Specific backend to use (optional)
            
        Returns:
            Job submission ID
        """
        if backend_name:
            if backend_name not in self.backends:
                raise ValueError(f"Backend '{backend_name}' not available")
            backend = self.backends[backend_name]
        else:
            # Select best backend automatically
            backend = self._select_best_backend(job)
        
        # Simulate job submission
        warnings.warn("Job submission is simulated - connect to real quantum providers")
        
        submission_id = f"submit_{job.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In real implementation, this would submit to actual quantum service
        print(f"Job {job.id} submitted to {backend.name} (ID: {submission_id})")
        
        return submission_id
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get current status of submitted job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Current job status
        """
        # Placeholder implementation
        # Real implementation would query quantum service APIs
        return JobStatus.PENDING
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel submitted job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation successful
        """
        # Placeholder implementation
        warnings.warn("Job cancellation is simulated")
        return True
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns:
            Queue status information
        """
        return {
            "total_jobs": len(self.job_queue),
            "completed_jobs": len(self.completed_jobs),
            "backends_available": len([b for b in self.backends.values() if b.availability]),
            "average_wait_time": self._calculate_average_wait_time()
        }
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load scheduler configuration."""
        default_config = {
            "providers": {
                "ibmq": {
                    "enabled": True,
                    "priority": 1,
                    "cost_weight": 1.0
                },
                "aws_braket": {
                    "enabled": False,
                    "priority": 2,
                    "cost_weight": 1.5
                }
            },
            "optimization": {
                "max_queue_time_minutes": 120,
                "cost_penalty_factor": 1.0,
                "time_penalty_factor": 1.0
            },
            "constraints": {
                "max_shots_per_job": 100000,
                "max_circuit_depth": 100
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                self._deep_merge(default_config, user_config)
            except Exception as e:
                warnings.warn(f"Failed to load scheduler config: {e}")
        
        return default_config
    
    def _initialize_backends(self):
        """Initialize available quantum backends."""
        # Mock backend data for demonstration
        self.backends = {
            "ibmq_qasm_simulator": BackendInfo(
                name="ibmq_qasm_simulator",
                provider="ibmq",
                num_qubits=32,
                queue_length=5,
                estimated_wait_time=2.0,
                cost_per_shot=0.0001,
                calibration_data={"fidelity": 0.99, "error_rate": 0.01}
            ),
            "ibmq_manhattan": BackendInfo(
                name="ibmq_manhattan",
                provider="ibmq", 
                num_qubits=65,
                queue_length=25,
                estimated_wait_time=45.0,
                cost_per_shot=0.001,
                calibration_data={"fidelity": 0.95, "error_rate": 0.05}
            ),
            "aws_sv1": BackendInfo(
                name="aws_sv1",
                provider="aws_braket",
                num_qubits=34,
                queue_length=0,
                estimated_wait_time=0.5,
                cost_per_shot=0.00075,
                calibration_data={"fidelity": 0.98, "error_rate": 0.02}
            )
        }
    
    def _calculate_priority_score(self, job: QuantumJob) -> float:
        """Calculate priority score for job scheduling."""
        base_score = job.priority.value * 100
        
        # Add urgency based on deadline
        if job.deadline:
            time_to_deadline = (job.deadline - datetime.now()).total_seconds() / 3600
            urgency_score = max(0, 100 - time_to_deadline)
            base_score += urgency_score
        
        # Add penalty for large resource requirements
        resource_penalty = min(50, job.shots / 1000)
        base_score -= resource_penalty
        
        return base_score
    
    def _get_available_backends(self, constraints: Dict[str, Any]) -> List[BackendInfo]:
        """Get list of available backends matching constraints."""
        available = []
        
        preferred_devices = constraints.get("preferred_devices", [])
        
        for backend in self.backends.values():
            if not backend.availability:
                continue
            
            # Check if backend is preferred
            if preferred_devices and backend.name not in preferred_devices:
                continue
            
            available.append(backend)
        
        return available
    
    def _optimize_for_cost(
        self,
        jobs: List[QuantumJob],
        backends: List[BackendInfo],
        constraints: Dict[str, Any]
    ) -> OptimizedSchedule:
        """Optimize schedule to minimize total cost."""
        schedule_entries = []
        current_time = datetime.now()
        
        # Sort jobs by priority and cost sensitivity
        sorted_jobs = sorted(jobs, key=lambda j: self._calculate_priority_score(j), reverse=True)
        
        for job in sorted_jobs:
            # Find cheapest suitable backend
            suitable_backends = [b for b in backends 
                               if self._is_backend_suitable(job, b)]
            
            if not suitable_backends:
                warnings.warn(f"No suitable backend found for job {job.id}")
                continue
            
            # Sort by cost per shot
            cheapest_backend = min(suitable_backends, key=lambda b: b.cost_per_shot)
            
            # Calculate timing
            estimated_cost = job.shots * cheapest_backend.cost_per_shot
            execution_time = self._estimate_execution_time(job, cheapest_backend)
            
            start_time = current_time + timedelta(minutes=cheapest_backend.estimated_wait_time)
            end_time = start_time + timedelta(seconds=execution_time)
            
            # Check budget constraint
            budget = constraints.get("budget")
            if budget and estimated_cost > budget:
                warnings.warn(f"Job {job.id} exceeds budget: ${estimated_cost:.2f} > ${budget:.2f}")
                continue
            
            entry = ScheduleEntry(
                job=job,
                backend=cheapest_backend,
                estimated_start_time=start_time,
                estimated_end_time=end_time,
                estimated_cost=estimated_cost,
                priority_score=self._calculate_priority_score(job)
            )
            
            schedule_entries.append(entry)
            current_time = end_time  # Update for next job
        
        return OptimizedSchedule(
            entries=schedule_entries,
            optimization_metadata={"goal": "minimize_cost"}
        )
    
    def _optimize_for_time(
        self,
        jobs: List[QuantumJob],
        backends: List[BackendInfo],
        constraints: Dict[str, Any]
    ) -> OptimizedSchedule:
        """Optimize schedule to minimize total execution time."""
        schedule_entries = []
        current_time = datetime.now()
        
        # Sort jobs by priority and time sensitivity
        sorted_jobs = sorted(jobs, key=lambda j: self._calculate_priority_score(j), reverse=True)
        
        for job in sorted_jobs:
            # Find fastest suitable backend
            suitable_backends = [b for b in backends 
                               if self._is_backend_suitable(job, b)]
            
            if not suitable_backends:
                warnings.warn(f"No suitable backend found for job {job.id}")
                continue
            
            # Sort by total time (wait + execution)
            fastest_backend = min(suitable_backends, 
                                key=lambda b: b.estimated_wait_time + 
                                self._estimate_execution_time(job, b) / 60)
            
            # Calculate timing and cost
            estimated_cost = job.shots * fastest_backend.cost_per_shot
            execution_time = self._estimate_execution_time(job, fastest_backend)
            
            start_time = current_time + timedelta(minutes=fastest_backend.estimated_wait_time)
            end_time = start_time + timedelta(seconds=execution_time)
            
            entry = ScheduleEntry(
                job=job,
                backend=fastest_backend,
                estimated_start_time=start_time,
                estimated_end_time=end_time,
                estimated_cost=estimated_cost,
                priority_score=self._calculate_priority_score(job)
            )
            
            schedule_entries.append(entry)
        
        return OptimizedSchedule(
            entries=schedule_entries,
            optimization_metadata={"goal": "minimize_time"}
        )
    
    def _is_backend_suitable(self, job: QuantumJob, backend: BackendInfo) -> bool:
        """Check if backend is suitable for job."""
        # Check backend requirements
        if job.backend_requirements:
            if backend.name not in job.backend_requirements:
                return False
        
        # Check if backend has enough qubits
        # This would require circuit analysis in real implementation
        required_qubits = getattr(job.circuit, 'num_qubits', 1)
        if required_qubits > backend.num_qubits:
            return False
        
        return True
    
    def _estimate_execution_time(self, job: QuantumJob, backend: BackendInfo) -> float:
        """Estimate job execution time in seconds."""
        # Simple estimation based on shots and backend characteristics
        base_time = 10  # seconds base execution time
        shot_time = job.shots * 0.001  # 1ms per shot
        queue_factor = 1 + (backend.queue_length / 10)  # Queue overhead
        
        return (base_time + shot_time) * queue_factor
    
    def _select_best_backend(self, job: QuantumJob) -> BackendInfo:
        """Select best backend for job based on optimization goal."""
        suitable_backends = [b for b in self.backends.values() 
                           if self._is_backend_suitable(job, b)]
        
        if not suitable_backends:
            raise ValueError(f"No suitable backend found for job {job.id}")
        
        if self.optimization_goal == "minimize_cost":
            return min(suitable_backends, key=lambda b: b.cost_per_shot)
        elif self.optimization_goal == "minimize_time":
            return min(suitable_backends, key=lambda b: b.estimated_wait_time)
        else:
            return suitable_backends[0]  # Default
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time across all backends."""
        if not self.backends:
            return 0.0
        
        total_wait = sum(b.estimated_wait_time for b in self.backends.values())
        return total_wait / len(self.backends)
    
    def _deep_merge(self, base: dict, update: dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# CLI interface for job scheduling
def main():
    """Main entry point for quantum job scheduler CLI."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Quantum job scheduler')
    parser.add_argument('--config', help='Scheduler configuration file')
    parser.add_argument('--goal', choices=['minimize_cost', 'minimize_time'], 
                       default='minimize_cost', help='Optimization goal')
    parser.add_argument('--jobs', help='Jobs specification file (JSON)')
    parser.add_argument('--constraints', help='Scheduling constraints (JSON)')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = QuantumJobScheduler(args.config, args.goal)
    
    if args.jobs:
        with open(args.jobs, 'r') as f:
            jobs = json.load(f)
        
        constraints = {}
        if args.constraints:
            with open(args.constraints, 'r') as f:
                constraints = json.load(f)
        
        # Optimize schedule
        schedule = scheduler.optimize_schedule(jobs, constraints)
        
        # Print results
        print(f"Optimized schedule ({args.goal}):")
        print(f"Total cost: ${schedule.total_cost:.2f}")
        print(f"Total time: {schedule.total_time_hours:.1f} hours")
        print(f"Jobs scheduled: {len(schedule.entries)}")
        print("\nDevice allocation:")
        for device, count in schedule.device_allocation.items():
            print(f"  {device}: {count} jobs")
    else:
        # Show queue status
        status = scheduler.get_queue_status()
        print("Quantum job scheduler status:")
        for key, value in status.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()