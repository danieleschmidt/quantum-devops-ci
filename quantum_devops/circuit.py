"""
QuantumCircuit: A simple, dependency-free quantum circuit representation.

Supports gates: H, CNOT, RZ, RX, MEASURE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


SUPPORTED_GATES = frozenset({"H", "CNOT", "RZ", "RX", "MEASURE"})
T_GATES = frozenset({"T", "T_DAG", "RZ"})  # RZ(pi/4) ≈ T gate in fault-tolerance counting


@dataclass
class Gate:
    """Represents a single quantum gate operation."""
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if isinstance(self.qubits, list):
            self.qubits = tuple(self.qubits)
        if isinstance(self.params, list):
            self.params = tuple(self.params)
        if self.name not in SUPPORTED_GATES:
            raise ValueError(
                f"Unsupported gate '{self.name}'. Supported: {sorted(SUPPORTED_GATES)}"
            )

    def __repr__(self) -> str:
        param_str = f"({', '.join(f'{p:.4f}' for p in self.params)})" if self.params else ""
        qubit_str = ", ".join(str(q) for q in self.qubits)
        return f"{self.name}{param_str} q[{qubit_str}]"

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

    @property
    def is_parametric(self) -> bool:
        return len(self.params) > 0


class QuantumCircuit:
    """
    A simple quantum circuit representation.

    Example:
        >>> qc = QuantumCircuit(2, name="Bell")
        >>> qc.h(0)
        >>> qc.cnot(0, 1)
        >>> qc.measure(0)
        >>> qc.measure(1)
        >>> qc.validate()
    """

    def __init__(self, num_qubits: int, name: str = "circuit"):
        if num_qubits < 1:
            raise ValueError("Circuit must have at least 1 qubit")
        self.num_qubits = num_qubits
        self.name = name
        self._gates: List[Gate] = []

    # ── Gate builders ────────────────────────────────────────────────────────

    def h(self, qubit: int) -> "QuantumCircuit":
        """Hadamard gate — creates superposition."""
        self._add_gate(Gate("H", (qubit,)))
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """CNOT (CX) gate — entangles two qubits."""
        if control == target:
            raise ValueError("CNOT control and target must differ")
        self._add_gate(Gate("CNOT", (control, target)))
        return self

    def rz(self, qubit: int, theta: float) -> "QuantumCircuit":
        """RZ rotation gate around Z axis by angle theta (radians)."""
        self._add_gate(Gate("RZ", (qubit,), (theta,)))
        return self

    def rx(self, qubit: int, theta: float) -> "QuantumCircuit":
        """RX rotation gate around X axis by angle theta (radians)."""
        self._add_gate(Gate("RX", (qubit,), (theta,)))
        return self

    def measure(self, qubit: Optional[int] = None) -> "QuantumCircuit":
        """Add measurement gate. If qubit is None, measure all qubits."""
        if qubit is None:
            for q in range(self.num_qubits):
                self._add_gate(Gate("MEASURE", (q,)))
        else:
            self._add_gate(Gate("MEASURE", (qubit,)))
        return self

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _add_gate(self, gate: Gate) -> None:
        """Add a gate after validating qubit indices."""
        for q in gate.qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range for {self.num_qubits}-qubit circuit"
                )
        self._gates.append(gate)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def gates(self) -> List[Gate]:
        return list(self._gates)

    @property
    def depth(self) -> int:
        """
        Circuit depth: number of time-steps (layers).
        Computed via a simple layer-scheduling algorithm.
        """
        if not self._gates:
            return 0
        qubit_time = [0] * self.num_qubits
        for gate in self._gates:
            if gate.name == "MEASURE":
                continue
            start = max(qubit_time[q] for q in gate.qubits)
            end = start + 1
            for q in gate.qubits:
                qubit_time[q] = end
        return max(qubit_time) if qubit_time else 0

    @property
    def gate_count(self) -> int:
        return sum(1 for g in self._gates if g.name != "MEASURE")

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(1 for g in self._gates if g.is_two_qubit and g.name != "MEASURE")

    @property
    def measurement_count(self) -> int:
        return sum(1 for g in self._gates if g.name == "MEASURE")

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """
        Validate circuit structure. Returns list of warnings/errors.
        Empty list means valid circuit.
        """
        issues = []

        measured_qubits = set()
        for i, gate in enumerate(self._gates):
            if gate.name == "MEASURE":
                for q in gate.qubits:
                    if q in measured_qubits:
                        issues.append(f"Gate {i}: qubit {q} measured more than once")
                    measured_qubits.add(q)
            else:
                for q in gate.qubits:
                    if q in measured_qubits:
                        issues.append(
                            f"Gate {i} ({gate.name}): qubit {q} used after measurement"
                        )

        if self.gate_count == 0:
            issues.append("Circuit has no gates (empty circuit)")

        if self.measurement_count == 0:
            issues.append("No measurements — circuit will produce no classical output")

        return issues

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(name='{self.name}', qubits={self.num_qubits}, "
            f"gates={self.gate_count}, depth={self.depth})"
        )

    def draw(self) -> str:
        """Return a simple ASCII representation of the circuit."""
        lines = [f"Circuit: {self.name} ({self.num_qubits} qubits)"]
        lines.append("─" * 50)
        for i, gate in enumerate(self._gates):
            lines.append(f"  {i:3d}: {gate}")
        lines.append("─" * 50)
        lines.append(
            f"  Depth: {self.depth}  Gates: {self.gate_count}  "
            f"2Q gates: {self.two_qubit_gate_count}"
        )
        return "\n".join(lines)
