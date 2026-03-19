"""
ResourceEstimator: Quantum resource estimation for CI/CD gate budget checks.

Estimates:
- Qubit count
- Circuit depth (time steps)
- Total gate count
- Two-qubit gate count (expensive on real hardware)
- T-gate count (critical for fault-tolerant quantum computing)
- Estimated QPU runtime (rough order-of-magnitude)

T-gates matter because in fault-tolerant quantum computing (FT-QC),
T-gates require magic state distillation, which is very expensive.
Tracking T-count is a standard FT-QC resource metric.

RZ(θ) gates are approximated as T-gates when |θ| ≈ π/4 (mod π/2),
otherwise they're treated as Clifford-equivalent (free in FT-QC).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .circuit import QuantumCircuit, Gate


# Rough gate times in microseconds (hardware ballpark, gate-model QPU)
GATE_TIMES_US = {
    "H": 0.05,
    "RX": 0.05,
    "RZ": 0.05,
    "CNOT": 0.3,
    "MEASURE": 1.0,
}
DEFAULT_GATE_TIME_US = 0.1

# T-gate threshold: RZ(θ) counts as T if |θ mod (π/2)| < ε
T_GATE_EPSILON = 0.01


def _is_t_gate(gate: Gate) -> bool:
    """Return True if this gate is equivalent to a T gate (RZ(π/4))."""
    if gate.name == "RZ" and gate.params:
        theta = abs(gate.params[0])
        # RZ(π/4) = T, RZ(π/2) = S, RZ(π) = Z (Clifford)
        # Only RZ(π/4 + k*π/2) for odd k counts as non-Clifford T-gate
        normalized = theta % (math.pi / 2)
        return abs(normalized - math.pi / 4) < T_GATE_EPSILON
    return False


def _is_clifford_rz(gate: Gate) -> bool:
    """RZ(k*π/2) for integer k are Clifford (free in FT-QC)."""
    if gate.name == "RZ" and gate.params:
        theta = abs(gate.params[0])
        normalized = theta % (math.pi / 2)
        return normalized < T_GATE_EPSILON or abs(normalized - math.pi / 2) < T_GATE_EPSILON
    return False


@dataclass
class ResourceReport:
    """Quantum resource estimation report for a single circuit."""
    circuit_name: str
    qubit_count: int
    depth: int
    gate_count: int
    two_qubit_gate_count: int
    t_gate_count: int
    clifford_gate_count: int
    measurement_count: int
    estimated_runtime_us: float
    gate_breakdown: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def fault_tolerance_overhead(self) -> str:
        """
        Rough FT overhead classification based on T-gate count.
        (Real FT overhead depends heavily on target error rate and code distance.)
        """
        if self.t_gate_count == 0:
            return "Clifford-only (zero distillation overhead)"
        if self.t_gate_count < 50:
            return "Low (few distillation rounds)"
        if self.t_gate_count < 500:
            return "Moderate (~{} distillation rounds)".format(self.t_gate_count * 7)
        return "High (~{} distillation rounds)".format(self.t_gate_count * 7)

    def summary(self) -> str:
        lines = [
            f"Resource Report: {self.circuit_name}",
            f"  Qubits:          {self.qubit_count}",
            f"  Circuit depth:   {self.depth}",
            f"  Total gates:     {self.gate_count}",
            f"  2Q gates:        {self.two_qubit_gate_count}",
            f"  T-gates:         {self.t_gate_count}",
            f"  Clifford gates:  {self.clifford_gate_count}",
            f"  Measurements:    {self.measurement_count}",
            f"  Est. runtime:    {self.estimated_runtime_us:.2f} µs",
            f"  FT overhead:     {self.fault_tolerance_overhead}",
        ]
        if self.gate_breakdown:
            lines.append("  Gate breakdown:")
            for name, count in sorted(self.gate_breakdown.items()):
                lines.append(f"    {name}: {count}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.circuit_name,
            "qubit_count": self.qubit_count,
            "depth": self.depth,
            "gate_count": self.gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "t_gate_count": self.t_gate_count,
            "clifford_gate_count": self.clifford_gate_count,
            "measurement_count": self.measurement_count,
            "estimated_runtime_us": self.estimated_runtime_us,
            "fault_tolerance_overhead": self.fault_tolerance_overhead,
            "gate_breakdown": self.gate_breakdown,
            "warnings": self.warnings,
        }


class ResourceEstimator:
    """
    Estimates quantum resource requirements for a QuantumCircuit.

    No external dependencies — pure stdlib analysis of the gate list.
    """

    def estimate(self, circuit: QuantumCircuit) -> ResourceReport:
        """Analyze a circuit and return a ResourceReport."""
        gate_breakdown: Dict[str, int] = {}
        t_count = 0
        clifford_count = 0
        runtime_us = 0.0
        two_q_count = 0

        for gate in circuit.gates:
            name = gate.name
            gate_breakdown[name] = gate_breakdown.get(name, 0) + 1

            if name != "MEASURE":
                runtime_us += GATE_TIMES_US.get(name, DEFAULT_GATE_TIME_US)

            if gate.is_two_qubit:
                two_q_count += 1

            if _is_t_gate(gate):
                t_count += 1
            elif name in ("H", "CNOT", "MEASURE") or _is_clifford_rz(gate):
                clifford_count += 1

        warnings = []
        if circuit.depth > 100:
            warnings.append(f"High circuit depth ({circuit.depth}) — consider optimization")
        if two_q_count > 20:
            warnings.append(
                f"High 2Q gate count ({two_q_count}) — significant noise exposure on NISQ hardware"
            )
        if t_count > 100:
            warnings.append(
                f"High T-gate count ({t_count}) — significant fault-tolerance overhead"
            )
        validation = circuit.validate()
        warnings.extend(validation)

        return ResourceReport(
            circuit_name=circuit.name,
            qubit_count=circuit.num_qubits,
            depth=circuit.depth,
            gate_count=circuit.gate_count,
            two_qubit_gate_count=two_q_count,
            t_gate_count=t_count,
            clifford_gate_count=clifford_count,
            measurement_count=circuit.measurement_count,
            estimated_runtime_us=runtime_us,
            gate_breakdown=gate_breakdown,
            warnings=warnings,
        )

    def estimate_suite(self, circuits: List[QuantumCircuit]) -> List[ResourceReport]:
        """Estimate resources for a list of circuits."""
        return [self.estimate(c) for c in circuits]

    def budget_check(
        self,
        circuit: QuantumCircuit,
        max_depth: Optional[int] = None,
        max_t_gates: Optional[int] = None,
        max_two_qubit_gates: Optional[int] = None,
    ) -> tuple[ResourceReport, List[str]]:
        """
        Check a circuit against resource budgets.

        Returns (report, violations) where violations is a list of
        budget violation messages (empty = within budget).
        """
        report = self.estimate(circuit)
        violations = []

        if max_depth is not None and report.depth > max_depth:
            violations.append(
                f"Depth {report.depth} exceeds budget {max_depth}"
            )
        if max_t_gates is not None and report.t_gate_count > max_t_gates:
            violations.append(
                f"T-gate count {report.t_gate_count} exceeds budget {max_t_gates}"
            )
        if max_two_qubit_gates is not None and report.two_qubit_gate_count > max_two_qubit_gates:
            violations.append(
                f"2Q gate count {report.two_qubit_gate_count} exceeds budget {max_two_qubit_gates}"
            )

        return report, violations
