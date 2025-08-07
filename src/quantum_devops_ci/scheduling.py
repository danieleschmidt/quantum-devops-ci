"""
Quantum job scheduling and resource management.

This module provides intelligent scheduling of quantum jobs across different
hardware providers, considering cost, time, and hardware constraints.
"""

import abc
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq
from enum import Enum
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("pyyaml not available - YAML configuration loading will be limited")


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
            self.backend_requirements = ["qasm_simulator"]
    
    @property
    def priority_score(self) -> int:
        """Get numeric priority score for scheduling."""
        return self.priority.value
    
    def estimated_cost(self, backend_cost_per_shot: float) -> float:
        """Estimate job cost based on shots and backend pricing."""
        return self.shots * backend_cost_per_shot
    
    def is_deadline_critical(self, current_time: datetime) -> bool:
        """Check if job deadline is approaching."""
        if not self.deadline:
            return False
        time_remaining = self.deadline - current_time
        return time_remaining.total_seconds() < 3600  # Less than 1 hour


@dataclass
class BackendInfo:
    """Information about quantum backend availability and capabilities."""
    name: str
    provider: str
    is_available: bool
    queue_length: int
    estimated_wait_time: int  # seconds
    cost_per_shot: float
    max_shots: int
    max_qubits: int
    gate_set: List[str]
    connectivity: Optional[List[List[int]]] = None
    error_rates: Optional[Dict[str, float]] = None
    calibration_time: Optional[datetime] = None
    
    @property
    def efficiency_score(self) -> float:
        """Calculate backend efficiency score for scheduling."""
        if not self.is_available:
            return 0.0
        
        # Lower cost and wait time = higher efficiency
        cost_factor = 1.0 / (self.cost_per_shot + 0.001)  # Avoid division by zero
        time_factor = 1.0 / (self.estimated_wait_time + 60)  # Minimum 1 minute
        
        return (cost_factor * time_factor) * 1000  # Scale for readability


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
    """Result of job scheduling optimization."""
    entries: List[ScheduleEntry]
    total_cost: float
    total_time_hours: float
    device_allocation: Dict[str, int]
    optimization_goal: str
    constraints_satisfied: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Recalculate totals based on entries
        self.total_cost = sum(entry.estimated_cost for entry in self.entries)
        if self.entries:
            start_time = min(entry.estimated_start_time for entry in self.entries)
            end_time = max(entry.estimated_end_time for entry in self.entries)
            self.total_time_hours = (end_time - start_time).total_seconds() / 3600
        else:
            self.total_time_hours = 0.0
        
        # Calculate device allocation
        device_counts = {}
        for entry in self.entries:
            device_name = entry.backend.name
            device_counts[device_name] = device_counts.get(device_name, 0) + 1
        self.device_allocation = device_counts
    
    def add_warning(self, message: str):
        """Add warning to schedule."""
        self.warnings.append(message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get scheduling summary."""
        return {
            'total_jobs': len(self.entries),
            'total_cost': self.total_cost,
            'total_time_hours': self.total_time_hours,
            'unique_backends': len(self.device_allocation),
            'average_cost_per_job': self.total_cost / len(self.entries) if self.entries else 0,
            'optimization_goal': self.optimization_goal,
            'constraints_satisfied': self.constraints_satisfied,
            'warnings_count': len(self.warnings)
        }


class QuantumJobScheduler:
    """Intelligent quantum job scheduler."""
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 optimization_goal: str = "minimize_cost"):
        """
        Initialize quantum job scheduler.
        
        Args:
            config_file: Path to configuration file
            optimization_goal: 'minimize_cost' or 'minimize_time'
        """
        self.optimization_goal = optimization_goal
        self.backends = {}
        self.job_queue = []
        self.completed_jobs = []
        self.config = {}
        
        if config_file:
            self.load_config(config_file)
        
        # Initialize default backends
        self._initialize_default_backends()
    
    def load_config(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            warnings.warn(f"Failed to load config: {e}")
    
    def _initialize_default_backends(self):
        """Initialize default backend information."""
        # Simulator backends (always available)
        self.backends['qasm_simulator'] = BackendInfo(
            name='qasm_simulator',
            provider='aer',
            is_available=True,
            queue_length=0,
            estimated_wait_time=0,
            cost_per_shot=0.0,
            max_shots=1000000,
            max_qubits=32,
            gate_set=['u1', 'u2', 'u3', 'cx']
        )
        
        self.backends['statevector_simulator'] = BackendInfo(
            name='statevector_simulator',
            provider='aer',
            is_available=True,
            queue_length=0,
            estimated_wait_time=0,
            cost_per_shot=0.0,
            max_shots=1000000,
            max_qubits=20,
            gate_set=['u1', 'u2', 'u3', 'cx']
        )
        
        # Mock hardware backends
        self.backends['ibmq_manhattan'] = BackendInfo(
            name='ibmq_manhattan',
            provider='ibmq',
            is_available=True,
            queue_length=15,
            estimated_wait_time=300,  # 5 minutes
            cost_per_shot=0.001,
            max_shots=8192,
            max_qubits=5,
            gate_set=['rz', 'sx', 'x', 'cx'],
            error_rates={'single_qubit': 0.001, 'two_qubit': 0.01}
        )
        
        self.backends['aws_sv1'] = BackendInfo(
            name='aws_sv1',
            provider='aws_braket',
            is_available=True,
            queue_length=0,
            estimated_wait_time=30,
            cost_per_shot=0.00075,
            max_shots=100000,
            max_qubits=34,
            gate_set=['rx', 'ry', 'rz', 'cnot']
        )
    
    def add_job(self, job: QuantumJob):
        """Add job to scheduling queue."""
        heapq.heappush(self.job_queue, (-job.priority_score, datetime.now(), job))
    
    def get_available_backends(self, requirements: Optional[List[str]] = None) -> List[BackendInfo]:
        """Get list of available backends matching requirements."""
        available = []
        
        for backend in self.backends.values():
            if not backend.is_available:
                continue
            
            if requirements:
                # Check if any requirement matches
                if not any(req in backend.name or req == backend.provider for req in requirements):
                    continue
            
            available.append(backend)
        
        return available
    
    def optimize_schedule(self, 
                        jobs: List[Dict[str, Any]], 
                        constraints: Optional[Dict[str, Any]] = None) -> OptimizedSchedule:
        """
        Optimize job scheduling based on goals and constraints.
        
        Args:
            jobs: List of job specifications
            constraints: Scheduling constraints (deadline, budget, etc.)
            
        Returns:
            OptimizedSchedule with job assignments
        """
        if constraints is None:
            constraints = {}
        
        # Convert job specifications to QuantumJob objects
        quantum_jobs = []
        for job_spec in jobs:
            priority_name = job_spec.get('priority', 'MEDIUM')
            if isinstance(priority_name, str):
                priority = Priority[priority_name.upper()]
            elif isinstance(priority_name, int):
                priority = Priority(priority_name)
            else:
                priority = Priority.MEDIUM
                
            job = QuantumJob(
                id=job_spec.get('id', f'job_{len(quantum_jobs)}'),
                circuit=job_spec.get('circuit', None),
                shots=job_spec.get('shots', 1000),
                priority=priority,
                backend_requirements=job_spec.get('backend_requirements'),
                cost_limit=job_spec.get('cost_limit')
            )
            quantum_jobs.append(job)
        
        # Initialize schedule
        schedule = OptimizedSchedule(
            entries=[],
            total_cost=0.0,
            total_time_hours=0.0,
            device_allocation={},
            optimization_goal=self.optimization_goal
        )
        
        # Sort jobs by priority
        quantum_jobs.sort(key=lambda j: j.priority_score, reverse=True)
        
        current_time = datetime.now()
        
        for job in quantum_jobs:
            # Find best backend for this job
            available_backends = self.get_available_backends(job.backend_requirements)
            
            if not available_backends:
                schedule.add_warning(f"No available backends for job {job.id}")
                continue
            
            # Select optimal backend based on goal
            if self.optimization_goal == "minimize_cost":
                best_backend = min(available_backends, key=lambda b: b.cost_per_shot)
            elif self.optimization_goal == "minimize_time":
                best_backend = min(available_backends, key=lambda b: b.estimated_wait_time)
            else:
                # Use efficiency score as default
                best_backend = max(available_backends, key=lambda b: b.efficiency_score)
            
            # Check constraints
            job_cost = job.estimated_cost(best_backend.cost_per_shot)
            
            if 'budget' in constraints and (schedule.total_cost + job_cost) > constraints['budget']:
                schedule.add_warning(f"Job {job.id} exceeds budget constraint")
                continue
            
            # Calculate timing
            execution_time_seconds = (job.shots * 0.001)  # Rough estimate
            scheduled_start_time = current_time + timedelta(seconds=best_backend.estimated_wait_time)
            scheduled_end_time = scheduled_start_time + timedelta(seconds=execution_time_seconds)
            
            # Create schedule entry
            entry = ScheduleEntry(
                job=job,
                backend=best_backend,
                estimated_start_time=scheduled_start_time,
                estimated_end_time=scheduled_end_time,
                estimated_cost=job_cost,
                priority_score=job.priority_score
            )
            
            schedule.entries.append(entry)
            
            # Update backend queue (simple simulation)
            best_backend.queue_length += 1
            best_backend.estimated_wait_time += 60  # Add 1 minute per job
        
        return schedule
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        total_jobs = len(self.job_queue)
        
        priority_counts = {}
        for _, _, job in self.job_queue:
            priority = job.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        backend_status = {}
        for name, backend in self.backends.items():
            backend_status[name] = {
                'available': backend.is_available,
                'queue_length': backend.queue_length,
                'wait_time_minutes': backend.estimated_wait_time // 60,
                'cost_per_shot': backend.cost_per_shot
            }
        
        return {
            'total_queued_jobs': total_jobs,
            'priority_distribution': priority_counts,
            'backend_status': backend_status,
            'completed_jobs': len(self.completed_jobs)
        }
    
    def update_backend_status(self, backend_name: str, **kwargs):
        """Update backend status information."""
        if backend_name in self.backends:
            backend = self.backends[backend_name]
            for key, value in kwargs.items():
                if hasattr(backend, key):
                    setattr(backend, key, value)
    
    def simulate_job_execution(self, job: QuantumJob, backend: BackendInfo) -> Dict[str, Any]:
        """Simulate job execution (for testing purposes)."""
        import time
        import random
        
        start_time = time.time()
        
        # Simulate execution time based on shots
        execution_time = (job.shots / 1000) * random.uniform(0.5, 2.0)
        time.sleep(min(execution_time, 0.1))  # Cap simulation time
        
        # Simulate success/failure
        success_rate = 0.95 if 'simulator' in backend.name else 0.90
        success = random.random() < success_rate
        
        end_time = time.time()
        
        result = {
            'job_id': job.id,
            'backend': backend.name,
            'status': JobStatus.COMPLETED if success else JobStatus.FAILED,
            'execution_time': end_time - start_time,
            'cost': job.estimated_cost(backend.cost_per_shot),
            'shots_completed': job.shots if success else job.shots // 2
        }
        
        return result


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