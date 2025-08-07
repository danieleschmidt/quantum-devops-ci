"""
Auto-scaling system for quantum DevOps CI/CD infrastructure.

This module provides intelligent scaling of compute resources based on
workload patterns, cost constraints, and performance requirements.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

from .exceptions import ResourceExhaustionError, ConfigurationError
from .monitoring import QuantumCIMonitor
from .cost import CostOptimizer


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    BACKEND_CONNECTIONS = "backend_connections"
    CACHE_SIZE = "cache_size"
    WORKER_POOL = "worker_pool"


@dataclass
class ScalingMetric:
    """Metric for scaling decisions."""
    name: str
    current_value: float
    target_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    
    def scaling_pressure(self) -> float:
        """Calculate scaling pressure (-1 to 1, negative means scale down)."""
        if self.current_value > self.threshold_up:
            return min(1.0, (self.current_value - self.threshold_up) / 
                      (self.target_value - self.threshold_up))
        elif self.current_value < self.threshold_down:
            return max(-1.0, (self.current_value - self.threshold_down) / 
                      (self.target_value - self.threshold_down))
        else:
            return 0.0


@dataclass
class ScalingPolicy:
    """Policy for auto-scaling a resource."""
    resource_type: ResourceType
    min_capacity: int
    max_capacity: int
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    scale_up_step: int = 1
    scale_down_step: int = 1
    metrics: List[str] = field(default_factory=list)
    cost_weight: float = 0.3  # How much cost factors into scaling decisions


@dataclass
class ScalingEvent:
    """Record of a scaling action."""
    timestamp: datetime
    resource_type: ResourceType
    action: ScalingAction
    old_capacity: int
    new_capacity: int
    trigger_metrics: Dict[str, float]
    cost_impact: float
    reason: str


class ResourceScaler:
    """Individual resource scaler."""
    
    def __init__(self, policy: ScalingPolicy, resource_manager):
        """Initialize resource scaler."""
        self.policy = policy
        self.resource_manager = resource_manager
        self.current_capacity = policy.min_capacity
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        self.scaling_history: List[ScalingEvent] = []
    
    def evaluate_scaling(self, metrics: Dict[str, ScalingMetric]) -> Optional[ScalingEvent]:
        """Evaluate whether scaling is needed."""
        now = datetime.now()
        
        # Check cooldown periods
        scale_up_ready = (now - self.last_scale_up).seconds >= self.policy.scale_up_cooldown_seconds
        scale_down_ready = (now - self.last_scale_down).seconds >= self.policy.scale_down_cooldown_seconds
        
        # Calculate scaling pressure from relevant metrics
        total_pressure = 0.0
        relevant_metrics = {}
        
        for metric_name in self.policy.metrics:
            if metric_name in metrics:
                metric = metrics[metric_name]
                pressure = metric.scaling_pressure()
                total_pressure += pressure * metric.weight
                relevant_metrics[metric_name] = metric.current_value
        
        # Normalize by number of metrics
        if self.policy.metrics:
            avg_pressure = total_pressure / len(self.policy.metrics)
        else:
            avg_pressure = 0.0
        
        # Determine action
        action = ScalingAction.NO_ACTION
        new_capacity = self.current_capacity
        
        if avg_pressure > 0.5 and scale_up_ready and self.current_capacity < self.policy.max_capacity:
            # Scale up
            action = ScalingAction.SCALE_UP
            new_capacity = min(
                self.policy.max_capacity,
                self.current_capacity + self.policy.scale_up_step
            )
            
        elif avg_pressure < -0.5 and scale_down_ready and self.current_capacity > self.policy.min_capacity:
            # Scale down
            action = ScalingAction.SCALE_DOWN
            new_capacity = max(
                self.policy.min_capacity,
                self.current_capacity - self.policy.scale_down_step
            )
        
        if action == ScalingAction.NO_ACTION:
            return None
        
        # Apply scaling action
        try:
            success = self.resource_manager.scale_resource(
                self.policy.resource_type,
                new_capacity
            )
            
            if success:
                # Calculate cost impact (simplified)
                cost_impact = (new_capacity - self.current_capacity) * 10.0  # $10 per unit
                
                event = ScalingEvent(
                    timestamp=now,
                    resource_type=self.policy.resource_type,
                    action=action,
                    old_capacity=self.current_capacity,
                    new_capacity=new_capacity,
                    trigger_metrics=relevant_metrics,
                    cost_impact=cost_impact,
                    reason=f"Average scaling pressure: {avg_pressure:.2f}"
                )
                
                # Update state
                self.current_capacity = new_capacity
                if action == ScalingAction.SCALE_UP:
                    self.last_scale_up = now
                else:
                    self.last_scale_down = now
                
                self.scaling_history.append(event)
                logging.info(f"Scaled {self.policy.resource_type.value} from {event.old_capacity} to {event.new_capacity}")
                
                return event
                
        except Exception as e:
            logging.error(f"Failed to scale {self.policy.resource_type.value}: {e}")
        
        return None


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, history_window_hours: int = 24):
        """Initialize predictive scaler."""
        self.history_window_hours = history_window_hours
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.pattern_cache = {}
        self.lock = threading.Lock()
    
    def add_metric_sample(self, metric_name: str, value: float):
        """Add metric sample to history."""
        with self.lock:
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            
            now = datetime.now()
            self.metric_history[metric_name].append((now, value))
            
            # Trim old data
            cutoff = now - timedelta(hours=self.history_window_hours)
            self.metric_history[metric_name] = [
                (ts, val) for ts, val in self.metric_history[metric_name]
                if ts > cutoff
            ]
    
    def predict_metric(self, metric_name: str, 
                      prediction_horizon_minutes: int = 30) -> Optional[float]:
        """Predict future metric value."""
        with self.lock:
            if metric_name not in self.metric_history:
                return None
            
            history = self.metric_history[metric_name]
            if len(history) < 5:  # Not enough data
                return None
            
            # Simple linear regression for prediction
            now = datetime.now()
            recent_history = [
                (ts, val) for ts, val in history
                if (now - ts).seconds <= 3600  # Last hour
            ]
            
            if len(recent_history) < 3:
                return None
            
            # Calculate trend
            timestamps = [(ts - recent_history[0][0]).total_seconds() 
                         for ts, _ in recent_history]
            values = [val for _, val in recent_history]
            
            # Simple linear trend calculation
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return values[-1]  # No trend, return last value
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict value at future time
            future_time = prediction_horizon_minutes * 60
            predicted_value = intercept + slope * future_time
            
            # Apply some bounds checking
            max_recent = max(values)
            min_recent = min(values)
            range_recent = max_recent - min_recent
            
            # Don't predict beyond 2x recent range
            predicted_value = max(min_recent - range_recent, 
                                min(max_recent + range_recent, predicted_value))
            
            return predicted_value
    
    def get_scaling_recommendation(self, metric_name: str, 
                                 current_policy: ScalingPolicy) -> Optional[ScalingAction]:
        """Get scaling recommendation based on prediction."""
        predicted_value = self.predict_metric(metric_name)
        
        if predicted_value is None:
            return None
        
        # Compare with thresholds
        if predicted_value > current_policy.scale_up_threshold:
            return ScalingAction.SCALE_UP
        elif predicted_value < current_policy.scale_down_threshold:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION


class AutoScalingManager:
    """Main auto-scaling manager."""
    
    def __init__(self,
                 resource_manager,
                 monitor: Optional[QuantumCIMonitor] = None,
                 cost_optimizer: Optional[CostOptimizer] = None):
        """Initialize auto-scaling manager."""
        self.resource_manager = resource_manager
        self.monitor = monitor
        self.cost_optimizer = cost_optimizer
        
        self.scalers: Dict[ResourceType, ResourceScaler] = {}
        self.predictive_scaler = PredictiveScaler()
        
        self.is_running = False
        self.scaling_thread = None
        self.evaluation_interval_seconds = 60  # Evaluate every minute
        
        # Metrics collection
        self.current_metrics: Dict[str, ScalingMetric] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        
        self._initialize_default_metrics()
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add scaling policy for a resource."""
        scaler = ResourceScaler(policy, self.resource_manager)
        self.scalers[policy.resource_type] = scaler
        logging.info(f"Added scaling policy for {policy.resource_type.value}")
    
    def add_metric_collector(self, metric_name: str, collector_func: Callable):
        """Add custom metric collector."""
        self.metric_collectors[metric_name] = collector_func
    
    def start(self):
        """Start auto-scaling manager."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logging.info("Auto-scaling manager started")
    
    def stop(self):
        """Stop auto-scaling manager."""
        self.is_running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
        logging.info("Auto-scaling manager stopped")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        status = {
            'is_running': self.is_running,
            'scalers': {},
            'current_metrics': {},
            'recent_events': []
        }
        
        # Scaler status
        for resource_type, scaler in self.scalers.items():
            status['scalers'][resource_type.value] = {
                'current_capacity': scaler.current_capacity,
                'min_capacity': scaler.policy.min_capacity,
                'max_capacity': scaler.policy.max_capacity,
                'last_scale_up': scaler.last_scale_up.isoformat() if scaler.last_scale_up != datetime.min else None,
                'last_scale_down': scaler.last_scale_down.isoformat() if scaler.last_scale_down != datetime.min else None
            }
        
        # Current metrics
        for name, metric in self.current_metrics.items():
            status['current_metrics'][name] = {
                'current_value': metric.current_value,
                'target_value': metric.target_value,
                'scaling_pressure': metric.scaling_pressure()
            }
        
        # Recent scaling events
        all_events = []
        for scaler in self.scalers.values():
            all_events.extend(scaler.scaling_history[-10:])  # Last 10 events per scaler
        
        all_events.sort(key=lambda x: x.timestamp, reverse=True)
        status['recent_events'] = [
            {
                'timestamp': event.timestamp.isoformat(),
                'resource_type': event.resource_type.value,
                'action': event.action.value,
                'old_capacity': event.old_capacity,
                'new_capacity': event.new_capacity,
                'cost_impact': event.cost_impact,
                'reason': event.reason
            }
            for event in all_events[:20]  # Last 20 events overall
        ]
        
        return status
    
    def force_scaling_evaluation(self) -> Dict[ResourceType, Optional[ScalingEvent]]:
        """Force immediate scaling evaluation."""
        self._collect_metrics()
        events = {}
        
        for resource_type, scaler in self.scalers.items():
            event = scaler.evaluate_scaling(self.current_metrics)
            events[resource_type] = event
        
        return events
    
    def _initialize_default_metrics(self):
        """Initialize default metric collectors."""
        def get_cpu_utilization():
            # Placeholder - would integrate with system monitoring
            import psutil
            return psutil.cpu_percent()
        
        def get_memory_utilization():
            import psutil
            return psutil.virtual_memory().percent
        
        def get_queue_length():
            # Would integrate with actual queue system
            if hasattr(self.resource_manager, 'get_queue_length'):
                return self.resource_manager.get_queue_length()
            return 0.0
        
        def get_average_response_time():
            # Would integrate with performance monitoring
            if self.monitor:
                # Get from monitoring system
                pass
            return 100.0  # ms
        
        self.metric_collectors.update({
            'cpu_utilization': get_cpu_utilization,
            'memory_utilization': get_memory_utilization,
            'queue_length': get_queue_length,
            'response_time_ms': get_average_response_time
        })
    
    def _scaling_loop(self):
        """Main scaling evaluation loop."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collect current metrics
                self._collect_metrics()
                
                # Update predictive scaler
                for name, metric in self.current_metrics.items():
                    self.predictive_scaler.add_metric_sample(name, metric.current_value)
                
                # Evaluate scaling for each resource
                scaling_events = []
                for resource_type, scaler in self.scalers.items():
                    event = scaler.evaluate_scaling(self.current_metrics)
                    if event:
                        scaling_events.append(event)
                
                # Log any scaling events
                if scaling_events:
                    total_cost_impact = sum(event.cost_impact for event in scaling_events)
                    logging.info(f"Executed {len(scaling_events)} scaling actions, cost impact: ${total_cost_impact:.2f}")
                
                # Sleep until next evaluation
                elapsed = time.time() - start_time
                sleep_time = max(0, self.evaluation_interval_seconds - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logging.error(f"Error in scaling loop: {e}")
                time.sleep(self.evaluation_interval_seconds)
    
    def _collect_metrics(self):
        """Collect all configured metrics."""
        for name, collector in self.metric_collectors.items():
            try:
                value = collector()
                
                # Create or update metric
                if name in self.current_metrics:
                    self.current_metrics[name].current_value = value
                else:
                    # Default metric configuration
                    self.current_metrics[name] = ScalingMetric(
                        name=name,
                        current_value=value,
                        target_value=self._get_default_target(name),
                        threshold_up=self._get_default_threshold_up(name),
                        threshold_down=self._get_default_threshold_down(name)
                    )
                    
            except Exception as e:
                logging.warning(f"Failed to collect metric {name}: {e}")
    
    def _get_default_target(self, metric_name: str) -> float:
        """Get default target value for metric."""
        defaults = {
            'cpu_utilization': 70.0,
            'memory_utilization': 70.0,
            'queue_length': 10.0,
            'response_time_ms': 200.0
        }
        return defaults.get(metric_name, 50.0)
    
    def _get_default_threshold_up(self, metric_name: str) -> float:
        """Get default scale-up threshold for metric."""
        defaults = {
            'cpu_utilization': 80.0,
            'memory_utilization': 85.0,
            'queue_length': 20.0,
            'response_time_ms': 500.0
        }
        return defaults.get(metric_name, 80.0)
    
    def _get_default_threshold_down(self, metric_name: str) -> float:
        """Get default scale-down threshold for metric."""
        defaults = {
            'cpu_utilization': 30.0,
            'memory_utilization': 40.0,
            'queue_length': 2.0,
            'response_time_ms': 50.0
        }
        return defaults.get(metric_name, 30.0)


# Utility functions

def create_default_scaling_policies() -> List[ScalingPolicy]:
    """Create default scaling policies."""
    return [
        ScalingPolicy(
            resource_type=ResourceType.THREAD_POOL,
            min_capacity=4,
            max_capacity=32,
            target_utilization=0.7,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            metrics=['cpu_utilization', 'queue_length']
        ),
        ScalingPolicy(
            resource_type=ResourceType.PROCESS_POOL,
            min_capacity=2,
            max_capacity=8,
            target_utilization=0.6,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            metrics=['cpu_utilization', 'memory_utilization']
        ),
        ScalingPolicy(
            resource_type=ResourceType.WORKER_POOL,
            min_capacity=2,
            max_capacity=16,
            target_utilization=0.75,
            scale_up_threshold=0.85,
            scale_down_threshold=0.4,
            metrics=['queue_length', 'response_time_ms']
        )
    ]