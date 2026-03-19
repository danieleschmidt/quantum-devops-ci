"""
NoisySimulator: Depolarizing noise model with Pauli error injection.

Uses a density-matrix-like approach via Pauli error sampling — no external
dependencies, stdlib only (random, math).

Model:
  After each gate, with probability `error_rate`, apply a random Pauli error
  (X, Y, or Z) to each qubit involved in the gate. This approximates
  a depolarizing channel: ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

For measurement, we track a classical shadow: each qubit is a probability
(0.0 = |0⟩, 1.0 = |1⟩, 0.5 = full superposition). This lets us efficiently
simulate noise effects on expectation values without full 2^n state vectors.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple
from .circuit import QuantumCircuit, Gate


class QubitState:
    """
    Compact qubit state as a Bloch-sphere amplitude pair (alpha, beta)
    where |ψ⟩ = alpha|0⟩ + beta|1⟩, normalized.
    """

    def __init__(self, alpha: complex = 1.0 + 0j, beta: complex = 0.0 + 0j):
        self.alpha = alpha
        self.beta = beta
        self._normalize()

    def _normalize(self):
        norm = math.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm > 1e-10:
            self.alpha /= norm
            self.beta /= norm

    @property
    def prob_zero(self) -> float:
        return abs(self.alpha) ** 2

    @property
    def prob_one(self) -> float:
        return abs(self.beta) ** 2

    def apply_h(self):
        """Hadamard: |0⟩ → |+⟩, |1⟩ → |−⟩"""
        a, b = self.alpha, self.beta
        s = 1.0 / math.sqrt(2)
        self.alpha = s * (a + b)
        self.beta = s * (a - b)

    def apply_rx(self, theta: float):
        """RX(θ): rotation around X axis"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        a, b = self.alpha, self.beta
        self.alpha = complex(c) * a - complex(0, s) * b
        self.beta = -complex(0, s) * a + complex(c) * b

    def apply_rz(self, theta: float):
        """RZ(θ): rotation around Z axis (phase shift)"""
        self.alpha *= complex(math.cos(-theta / 2), math.sin(-theta / 2))
        self.beta *= complex(math.cos(theta / 2), math.sin(theta / 2))

    def apply_pauli_x(self):
        """Pauli X (bit flip): |0⟩↔|1⟩"""
        self.alpha, self.beta = self.beta, self.alpha

    def apply_pauli_z(self):
        """Pauli Z (phase flip): |1⟩ → -|1⟩"""
        self.beta = -self.beta

    def apply_pauli_y(self):
        """Pauli Y: iZ·X"""
        self.apply_pauli_x()
        self.apply_pauli_z()

    def measure(self, rng: random.Random) -> int:
        """Collapse qubit; return 0 or 1."""
        outcome = 1 if rng.random() < self.prob_one else 0
        if outcome == 0:
            self.alpha = 1.0 + 0j
            self.beta = 0.0 + 0j
        else:
            self.alpha = 0.0 + 0j
            self.beta = 1.0 + 0j
        return outcome

    def fidelity_to_zero(self) -> float:
        """Fidelity with |0⟩ state."""
        return self.prob_zero

    def __repr__(self) -> str:
        return f"QubitState({self.prob_zero:.3f}|0⟩ + {self.prob_one:.3f}|1⟩)"


class NoisySimulator:
    """
    Simulates a QuantumCircuit with depolarizing noise.

    Each gate application probabilistically injects Pauli errors
    (X, Y, Z) on the involved qubits with probability `error_rate`.

    CNOT is modeled as a correlated two-qubit gate: both qubits may
    receive independent errors.

    Args:
        error_rate: Per-gate depolarizing error probability [0.0, 1.0]
        seed: Optional RNG seed for reproducibility
    """

    PAULI_ERRORS = ["X", "Y", "Z"]

    def __init__(self, error_rate: float = 0.0, seed: Optional[int] = None):
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
        self.error_rate = error_rate
        self._rng = random.Random(seed)

    def run(self, circuit: QuantumCircuit, shots: int = 1) -> List[Dict[int, int]]:
        """
        Execute circuit `shots` times, return list of measurement dicts.

        Returns:
            List of {qubit_index: measurement_outcome} dicts, one per shot.
        """
        results = []
        for _ in range(shots):
            results.append(self._run_once(circuit))
        return results

    def run_and_aggregate(
        self, circuit: QuantumCircuit, shots: int = 100
    ) -> Dict[str, object]:
        """
        Run `shots` times and return aggregated statistics.

        Returns dict with:
            - counts: {bitstring: count}
            - fidelity: estimated fidelity (fraction of shots matching ideal)
            - error_rate: the configured error rate
            - shots: number of shots
        """
        shot_results = self.run(circuit, shots)

        # Build counts
        counts: Dict[str, int] = {}
        measured_qubits = sorted(
            set(q for r in shot_results for q in r.keys())
        )

        for result in shot_results:
            if not result:
                bitstring = "no_meas"
            else:
                bitstring = "".join(str(result.get(q, 0)) for q in measured_qubits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Fidelity: for a circuit with no noise, estimate the ideal output
        # We compute it as the fraction of shots matching the most-likely ideal state
        ideal_counts = self._get_ideal_counts(circuit, shots)
        fidelity = self._compute_fidelity(counts, ideal_counts, shots)

        return {
            "counts": counts,
            "fidelity": fidelity,
            "error_rate": self.error_rate,
            "shots": shots,
            "measured_qubits": measured_qubits,
        }

    def _run_once(self, circuit: QuantumCircuit) -> Dict[int, int]:
        """Single shot execution. Returns {qubit: outcome} for measured qubits."""
        qubits = [QubitState() for _ in range(circuit.num_qubits)]
        measurements: Dict[int, int] = {}

        for gate in circuit.gates:
            self._apply_gate(gate, qubits)
            self._maybe_inject_error(gate, qubits)

            if gate.name == "MEASURE":
                q = gate.qubits[0]
                measurements[q] = qubits[q].measure(self._rng)

        return measurements

    def _apply_gate(self, gate: Gate, qubits: List[QubitState]) -> None:
        """Apply gate to qubit states (ideal, no noise)."""
        name = gate.name
        if name == "H":
            qubits[gate.qubits[0]].apply_h()
        elif name == "RX":
            qubits[gate.qubits[0]].apply_rx(gate.params[0])
        elif name == "RZ":
            qubits[gate.qubits[0]].apply_rz(gate.params[0])
        elif name == "CNOT":
            ctrl, tgt = gate.qubits
            # If control is |1⟩ (prob > 0.5), flip target
            # Approximate CNOT for product states
            p_ctrl_one = qubits[ctrl].prob_one
            # Apply partial X to target proportional to control being |1⟩
            if self._rng.random() < p_ctrl_one:
                qubits[tgt].apply_pauli_x()
        elif name == "MEASURE":
            pass  # Handled in caller

    def _maybe_inject_error(self, gate: Gate, qubits: List[QubitState]) -> None:
        """Probabilistically inject depolarizing errors after gate."""
        if self.error_rate == 0.0 or gate.name == "MEASURE":
            return
        for q in gate.qubits:
            if self._rng.random() < self.error_rate:
                error = self._rng.choice(self.PAULI_ERRORS)
                if error == "X":
                    qubits[q].apply_pauli_x()
                elif error == "Y":
                    qubits[q].apply_pauli_y()
                else:
                    qubits[q].apply_pauli_z()

    def _get_ideal_counts(
        self, circuit: QuantumCircuit, shots: int
    ) -> Dict[str, int]:
        """Run the same circuit with zero noise to get ideal distribution."""
        ideal_sim = NoisySimulator(error_rate=0.0, seed=42)
        ideal_results = ideal_sim.run(circuit, shots)

        measured_qubits = sorted(
            set(q for r in ideal_results for q in r.keys())
        )
        counts: Dict[str, int] = {}
        for result in ideal_results:
            bitstring = "".join(str(result.get(q, 0)) for q in measured_qubits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def _compute_fidelity(
        self,
        noisy_counts: Dict[str, int],
        ideal_counts: Dict[str, int],
        shots: int,
    ) -> float:
        """
        Compute estimated fidelity as overlap between noisy and ideal distributions.
        F = Σ_x sqrt(P_ideal(x) * P_noisy(x))  (Bhattacharyya coefficient)
        """
        all_keys = set(noisy_counts) | set(ideal_counts)
        if not all_keys:
            return 1.0
        fidelity = 0.0
        for key in all_keys:
            p_ideal = ideal_counts.get(key, 0) / shots
            p_noisy = noisy_counts.get(key, 0) / shots
            fidelity += math.sqrt(p_ideal * p_noisy)
        return round(fidelity, 4)
