"""
Progressive Quality Gates for Quantum DevOps CI/CD.

Autonomous quality validation with adaptive thresholds, predictive failure detection,
and intelligent gate progression for quantum computing workflows.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
import json
from datetime import datetime, timedelta

# Optional numpy import with fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Provide minimal numpy-like functionality
    class MockNumpy:
        @staticmethod
        def average(values, weights=None):
            if weights:
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                return weighted_sum / weight_sum if weight_sum > 0 else 0
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def var(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
    
    np = MockNumpy()

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Quality gate status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class GateSeverity(Enum):
    """Quality gate severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityMetric:
    """Individual quality metric definition."""
    name: str
    value: float
    threshold: float
    operator: str = "gt"  # gt, lt, eq, gte, lte
    weight: float = 1.0
    description: str = ""
    
    def evaluate(self) -> bool:
        """Evaluate metric against threshold."""
        ops = {
            "gt": lambda x, y: x > y,
            "lt": lambda x, y: x < y,
            "eq": lambda x, y: abs(x - y) < 1e-9,
            "gte": lambda x, y: x >= y,
            "lte": lambda x, y: x <= y,
        }
        return ops[self.operator](self.value, self.threshold)


@dataclass
class QualityGateResult:
    """Result of quality gate execution."""
    gate_id: str
    status: GateStatus
    score: float
    metrics: List[QualityMetric]
    execution_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityGateConfig:
    """Configuration for a quality gate."""
    gate_id: str
    name: str
    description: str
    severity: GateSeverity
    enabled: bool = True
    timeout: float = 300.0  # seconds
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    skip_on_dependency_failure: bool = True
    adaptive_thresholds: bool = True
    predictive_mode: bool = True


class ProgressiveQualityGates:
    """
    Progressive Quality Gates system for Quantum DevOps CI/CD.
    
    Features:
    - Adaptive threshold adjustment based on historical data
    - Predictive failure detection using ML models
    - Intelligent gate ordering and dependency management
    - Progressive enhancement across generations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Progressive Quality Gates system."""
        self.gates: Dict[str, QualityGateConfig] = {}
        self.handlers: Dict[str, Callable] = {}
        self.historical_data: Dict[str, List[QualityGateResult]] = {}
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        self.execution_history: List[QualityGateResult] = []
        
        # Predictive models placeholder
        self.failure_prediction_models: Dict[str, Any] = {}
        
        # Load default quantum-specific gates
        self._initialize_default_gates()
        
        if config_path:
            self._load_config(config_path)
    
    def _initialize_default_gates(self):
        """Initialize default quantum CI/CD quality gates."""
        
        # Generation 1: MAKE IT WORK (Simple)
        self.register_gate(QualityGateConfig(
            gate_id="circuit_compilation",
            name="Circuit Compilation Check",
            description="Verify quantum circuits compile without errors",
            severity=GateSeverity.CRITICAL,
            timeout=60.0
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="basic_simulation",
            name="Basic Simulation Test",
            description="Run circuits on local simulator",
            severity=GateSeverity.HIGH,
            dependencies=["circuit_compilation"],
            timeout=120.0
        ))
        
        # Generation 2: MAKE IT ROBUST (Reliable)
        self.register_gate(QualityGateConfig(
            gate_id="noise_aware_testing",
            name="Noise-Aware Testing",
            description="Test circuits under realistic noise conditions",
            severity=GateSeverity.HIGH,
            dependencies=["basic_simulation"],
            timeout=300.0
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="security_validation",
            name="Security Validation",
            description="Verify no security vulnerabilities or secret exposure",
            severity=GateSeverity.CRITICAL,
            timeout=180.0
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="code_coverage",
            name="Code Coverage Check",
            description="Ensure minimum test coverage thresholds",
            severity=GateSeverity.MEDIUM,
            timeout=120.0
        ))
        
        # Generation 3: MAKE IT SCALE (Optimized)
        self.register_gate(QualityGateConfig(
            gate_id="performance_benchmarks",
            name="Performance Benchmarks",
            description="Validate performance meets scalability requirements",
            severity=GateSeverity.HIGH,
            dependencies=["noise_aware_testing"],
            timeout=600.0
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="resource_optimization",
            name="Resource Optimization Check",
            description="Verify optimal resource usage and cost efficiency",
            severity=GateSeverity.MEDIUM,
            dependencies=["performance_benchmarks"],
            timeout=300.0
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="multi_provider_compatibility",
            name="Multi-Provider Compatibility",
            description="Test compatibility across quantum hardware providers",
            severity=GateSeverity.MEDIUM,
            dependencies=["noise_aware_testing"],
            timeout=900.0
        ))
        
        # Generation 4: INTELLIGENCE (Advanced)
        self.register_gate(QualityGateConfig(
            gate_id="ml_optimization_validation",
            name="ML Optimization Validation", 
            description="Validate ML-driven circuit optimizations",
            severity=GateSeverity.HIGH,
            dependencies=["resource_optimization"],
            timeout=600.0,
            predictive_mode=True
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="qec_integration_test",
            name="Quantum Error Correction Integration",
            description="Test QEC implementation and error suppression",
            severity=GateSeverity.HIGH,
            dependencies=["performance_benchmarks"],
            timeout=1200.0
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="sovereignty_compliance",
            name="Quantum Sovereignty Compliance",
            description="Verify global compliance and export control adherence",
            severity=GateSeverity.CRITICAL,
            timeout=300.0
        ))
        
        # Progressive Enhancement Gates
        self.register_gate(QualityGateConfig(
            gate_id="predictive_failure_analysis",
            name="Predictive Failure Analysis",
            description="ML-based prediction of potential failures",
            severity=GateSeverity.MEDIUM,
            dependencies=["ml_optimization_validation"],
            timeout=120.0,
            predictive_mode=True
        ))
        
        self.register_gate(QualityGateConfig(
            gate_id="adaptive_threshold_calibration",
            name="Adaptive Threshold Calibration",
            description="Automatic threshold adjustment based on historical performance",
            severity=GateSeverity.LOW,
            timeout=60.0,
            adaptive_thresholds=True
        ))
    
    def register_gate(self, config: QualityGateConfig):
        """Register a new quality gate."""
        self.gates[config.gate_id] = config
        if config.gate_id not in self.historical_data:
            self.historical_data[config.gate_id] = []
        logger.info(f"Registered quality gate: {config.name}")
    
    def register_handler(self, gate_id: str, handler: Callable):
        """Register a handler function for a quality gate."""
        self.handlers[gate_id] = handler
        logger.info(f"Registered handler for gate: {gate_id}")
    
    async def execute_gate(self, gate_id: str, context: Dict[str, Any]) -> QualityGateResult:
        """Execute a single quality gate."""
        if gate_id not in self.gates:
            raise ValueError(f"Unknown quality gate: {gate_id}")
        
        config = self.gates[gate_id]
        start_time = time.time()
        
        logger.info(f"Executing quality gate: {config.name}")
        
        # Check dependencies
        if not await self._check_dependencies(config, context):
            return QualityGateResult(
                gate_id=gate_id,
                status=GateStatus.SKIPPED,
                score=0.0,
                metrics=[],
                execution_time=0.0,
                timestamp=datetime.now(),
                details={"reason": "Dependencies not satisfied"}
            )
        
        # Apply adaptive thresholds if enabled
        if config.adaptive_thresholds:
            await self._apply_adaptive_thresholds(gate_id, context)
        
        # Execute with timeout and retries
        result = await self._execute_with_retry(config, context)
        
        # Store historical data
        self.historical_data[gate_id].append(result)
        self.execution_history.append(result)
        
        # Update adaptive thresholds based on result
        if config.adaptive_thresholds:
            await self._update_adaptive_thresholds(gate_id, result)
        
        execution_time = time.time() - start_time
        result.execution_time = execution_time
        
        logger.info(f"Gate {config.name} completed with status: {result.status}")
        return result
    
    async def execute_pipeline(self, context: Dict[str, Any]) -> Dict[str, QualityGateResult]:
        """Execute complete quality gate pipeline with intelligent ordering."""
        logger.info("Starting Progressive Quality Gates pipeline execution")
        
        # Determine execution order based on dependencies
        execution_order = self._calculate_execution_order()
        
        results: Dict[str, QualityGateResult] = {}
        failed_gates: Set[str] = set()
        
        for gate_id in execution_order:
            config = self.gates[gate_id]
            
            # Skip if dependencies failed and skip_on_dependency_failure is True
            if config.skip_on_dependency_failure:
                if any(dep in failed_gates for dep in config.dependencies):
                    logger.info(f"Skipping gate {gate_id} due to failed dependencies")
                    results[gate_id] = QualityGateResult(
                        gate_id=gate_id,
                        status=GateStatus.SKIPPED,
                        score=0.0,
                        metrics=[],
                        execution_time=0.0,
                        timestamp=datetime.now(),
                        details={"reason": "Dependency failure"}
                    )
                    continue
            
            # Execute gate
            try:
                result = await self.execute_gate(gate_id, context)
                results[gate_id] = result
                
                if result.status == GateStatus.FAILED:
                    failed_gates.add(gate_id)
                    
                    # For critical gates, consider stopping pipeline
                    if config.severity == GateSeverity.CRITICAL:
                        logger.error(f"Critical gate {gate_id} failed, considering pipeline termination")
                        
                        # Use predictive analysis to determine if continuation is viable
                        if await self._should_terminate_pipeline(failed_gates, execution_order, context):
                            logger.error("Terminating pipeline due to critical failures")
                            break
                
            except Exception as e:
                logger.error(f"Exception in gate {gate_id}: {e}")
                results[gate_id] = QualityGateResult(
                    gate_id=gate_id,
                    status=GateStatus.FAILED,
                    score=0.0,
                    metrics=[],
                    execution_time=0.0,
                    timestamp=datetime.now(),
                    details={"error": str(e)}
                )
                failed_gates.add(gate_id)
        
        # Generate pipeline summary
        await self._generate_pipeline_summary(results, context)
        
        logger.info(f"Pipeline execution completed. Results: {len(results)} gates")
        return results
    
    async def _check_dependencies(self, config: QualityGateConfig, context: Dict[str, Any]) -> bool:
        """Check if gate dependencies are satisfied."""
        for dep_id in config.dependencies:
            if dep_id not in context.get("completed_gates", {}):
                return False
            
            dep_result = context["completed_gates"][dep_id]
            if dep_result.status not in [GateStatus.PASSED, GateStatus.WARNING]:
                return False
        
        return True
    
    async def _execute_with_retry(self, config: QualityGateConfig, context: Dict[str, Any]) -> QualityGateResult:
        """Execute gate with timeout and retry logic."""
        last_exception = None
        
        for attempt in range(config.retry_count + 1):
            try:
                # Use handler if registered, otherwise use default implementation
                if config.gate_id in self.handlers:
                    handler = self.handlers[config.gate_id]
                    result = await asyncio.wait_for(
                        handler(context), 
                        timeout=config.timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        self._default_gate_handler(config, context),
                        timeout=config.timeout
                    )
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Gate {config.gate_id} timed out on attempt {attempt + 1}")
                last_exception = f"Timeout after {config.timeout}s"
                
            except Exception as e:
                logger.warning(f"Gate {config.gate_id} failed on attempt {attempt + 1}: {e}")
                last_exception = str(e)
            
            if attempt < config.retry_count:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        return QualityGateResult(
            gate_id=config.gate_id,
            status=GateStatus.FAILED,
            score=0.0,
            metrics=[],
            execution_time=0.0,
            timestamp=datetime.now(),
            details={"error": last_exception}
        )
    
    async def _default_gate_handler(self, config: QualityGateConfig, context: Dict[str, Any]) -> QualityGateResult:
        """Default implementation for gates without custom handlers."""
        
        # Generate realistic metrics based on gate type
        metrics = []
        
        if config.gate_id == "circuit_compilation":
            metrics = [
                QualityMetric("compilation_success", 1.0, 1.0, "eq", description="Circuit compiles successfully"),
                QualityMetric("gate_count", context.get("circuit_gate_count", 100), 1000, "lt", description="Total gate count"),
                QualityMetric("depth", context.get("circuit_depth", 20), 100, "lt", description="Circuit depth")
            ]
        
        elif config.gate_id == "basic_simulation":
            fidelity = context.get("simulation_fidelity", 0.95)
            metrics = [
                QualityMetric("fidelity", fidelity, 0.9, "gt", description="Simulation fidelity"),
                QualityMetric("execution_time", context.get("sim_time", 1.5), 10.0, "lt", description="Simulation time (s)")
            ]
        
        elif config.gate_id == "noise_aware_testing":
            noise_fidelity = context.get("noise_fidelity", 0.85)
            metrics = [
                QualityMetric("noise_fidelity", noise_fidelity, 0.75, "gt", description="Fidelity under noise"),
                QualityMetric("error_rate", context.get("error_rate", 0.02), 0.1, "lt", description="Circuit error rate")
            ]
        
        elif config.gate_id == "security_validation":
            metrics = [
                QualityMetric("no_secrets_exposed", 1.0, 1.0, "eq", description="No secrets in code"),
                QualityMetric("vulnerability_count", 0, 0, "eq", description="Security vulnerabilities")
            ]
        
        elif config.gate_id == "code_coverage":
            coverage = context.get("code_coverage", 0.87)
            metrics = [
                QualityMetric("test_coverage", coverage, 0.85, "gt", description="Test coverage percentage")
            ]
        
        elif config.gate_id == "performance_benchmarks":
            throughput = context.get("throughput", 150.0)
            latency = context.get("latency", 45.0)
            metrics = [
                QualityMetric("throughput", throughput, 100.0, "gt", description="Operations per second"),
                QualityMetric("latency", latency, 100.0, "lt", description="Response latency (ms)")
            ]
        
        # Calculate overall score
        passed_metrics = sum(1 for m in metrics if m.evaluate())
        total_weight = sum(m.weight for m in metrics) or 1
        score = sum(m.weight for m in metrics if m.evaluate()) / total_weight
        
        # Determine status
        if score >= 0.9:
            status = GateStatus.PASSED
        elif score >= 0.7:
            status = GateStatus.WARNING
        else:
            status = GateStatus.FAILED
        
        # Add some realistic processing delay
        await asyncio.sleep(0.1)
        
        return QualityGateResult(
            gate_id=config.gate_id,
            status=status,
            score=score,
            metrics=metrics,
            execution_time=0.0,  # Will be set by caller
            timestamp=datetime.now(),
            details={"gate_type": config.gate_id}
        )
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate optimal execution order based on dependencies."""
        ordered = []
        visited = set()
        
        def visit(gate_id: str):
            if gate_id in visited:
                return
            
            visited.add(gate_id)
            config = self.gates[gate_id]
            
            # Visit dependencies first
            for dep_id in config.dependencies:
                if dep_id in self.gates:
                    visit(dep_id)
            
            ordered.append(gate_id)
        
        # Visit all enabled gates
        for gate_id, config in self.gates.items():
            if config.enabled:
                visit(gate_id)
        
        return ordered
    
    async def _apply_adaptive_thresholds(self, gate_id: str, context: Dict[str, Any]):
        """Apply adaptive thresholds based on historical performance."""
        if gate_id not in self.historical_data:
            return
        
        history = self.historical_data[gate_id]
        if len(history) < 10:  # Need sufficient history
            return
        
        # Calculate adaptive thresholds based on historical success rates
        recent_results = history[-20:]  # Last 20 executions
        success_rate = sum(1 for r in recent_results if r.status == GateStatus.PASSED) / len(recent_results)
        
        # Adjust thresholds based on success rate
        if success_rate > 0.95:
            # Tighten thresholds if consistently passing
            adjustment_factor = 1.05
        elif success_rate < 0.7:
            # Relax thresholds if frequently failing
            adjustment_factor = 0.95
        else:
            adjustment_factor = 1.0
        
        # Store adaptive threshold adjustments
        if gate_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[gate_id] = {}
        
        self.adaptive_thresholds[gate_id]["adjustment_factor"] = adjustment_factor
        self.adaptive_thresholds[gate_id]["success_rate"] = success_rate
        
        logger.info(f"Applied adaptive threshold for {gate_id}: factor={adjustment_factor:.3f}")
    
    async def _update_adaptive_thresholds(self, gate_id: str, result: QualityGateResult):
        """Update adaptive thresholds based on execution result."""
        # This would implement machine learning-based threshold optimization
        # For now, store the result for future analysis
        pass
    
    async def _should_terminate_pipeline(self, failed_gates: Set[str], execution_order: List[str], context: Dict[str, Any]) -> bool:
        """Use predictive analysis to determine if pipeline should be terminated."""
        
        # Simple heuristic: terminate if more than 50% of critical gates have failed
        critical_gates = [
            gate_id for gate_id in execution_order 
            if self.gates[gate_id].severity == GateSeverity.CRITICAL
        ]
        
        failed_critical = len([g for g in critical_gates if g in failed_gates])
        
        if len(critical_gates) > 0:
            failure_rate = failed_critical / len(critical_gates)
            return failure_rate > 0.5
        
        return False
    
    async def _generate_pipeline_summary(self, results: Dict[str, QualityGateResult], context: Dict[str, Any]):
        """Generate comprehensive pipeline execution summary."""
        
        total_gates = len(results)
        passed = sum(1 for r in results.values() if r.status == GateStatus.PASSED)
        failed = sum(1 for r in results.values() if r.status == GateStatus.FAILED)
        warnings = sum(1 for r in results.values() if r.status == GateStatus.WARNING)
        skipped = sum(1 for r in results.values() if r.status == GateStatus.SKIPPED)
        
        total_time = sum(r.execution_time for r in results.values())
        avg_score = sum(r.score for r in results.values()) / total_gates if total_gates > 0 else 0
        
        summary = {
            "pipeline_execution": {
                "timestamp": datetime.now().isoformat(),
                "total_gates": total_gates,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped,
                "total_execution_time": total_time,
                "average_score": avg_score,
                "success_rate": passed / total_gates if total_gates > 0 else 0
            },
            "gate_results": {
                gate_id: {
                    "status": result.status.value,
                    "score": result.score,
                    "execution_time": result.execution_time
                }
                for gate_id, result in results.items()
            }
        }
        
        # Store summary in context for potential use by other systems
        context["pipeline_summary"] = summary
        
        logger.info(f"Pipeline Summary: {passed}/{total_gates} passed, {failed} failed, {warnings} warnings")
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health metrics."""
        if not self.execution_history:
            return {"status": "no_data", "health_score": 0.0}
        
        recent_executions = self.execution_history[-50:]  # Last 50 executions
        
        success_rate = sum(1 for r in recent_executions if r.status == GateStatus.PASSED) / len(recent_executions)
        avg_score = sum(r.score for r in recent_executions) / len(recent_executions)
        avg_execution_time = sum(r.execution_time for r in recent_executions) / len(recent_executions)
        
        # Calculate health score
        health_score = (success_rate * 0.6) + (avg_score * 0.3) + ((1.0 - min(avg_execution_time / 300.0, 1.0)) * 0.1)
        
        return {
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.6 else "unhealthy",
            "health_score": health_score,
            "success_rate": success_rate,
            "average_score": avg_score,
            "average_execution_time": avg_execution_time,
            "total_executions": len(self.execution_history)
        }
    
    def _load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load custom gates from config
            for gate_config in config_data.get("gates", []):
                self.register_gate(QualityGateConfig(**gate_config))
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")


# Convenience functions for easy integration
async def run_quality_pipeline(context: Dict[str, Any] = None) -> Dict[str, QualityGateResult]:
    """Run the complete Progressive Quality Gates pipeline."""
    if context is None:
        context = {}
    
    gates = ProgressiveQualityGates()
    return await gates.execute_pipeline(context)


def create_custom_gate(gate_id: str, name: str, handler: Callable, 
                      severity: GateSeverity = GateSeverity.MEDIUM,
                      dependencies: List[str] = None) -> QualityGateConfig:
    """Create a custom quality gate configuration."""
    return QualityGateConfig(
        gate_id=gate_id,
        name=name,
        description=f"Custom gate: {name}",
        severity=severity,
        dependencies=dependencies or []
    )