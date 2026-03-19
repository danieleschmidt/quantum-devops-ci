"""
Example quantum circuits for testing and demonstration.

Includes:
- Bell state (2 qubits, entanglement)
- GHZ state (3 qubits, multi-qubit entanglement)
- QAOA-like ansatz (4 qubits, variational circuit)
"""

import math
from quantum_devops import QuantumCircuit


def bell_state() -> QuantumCircuit:
    """
    Bell state preparation: |Φ+⟩ = (|00⟩ + |11⟩) / √2

    The simplest entangled state — a foundational circuit in quantum information.
    Expected output: 50% |00⟩, 50% |11⟩
    """
    qc = QuantumCircuit(2, name="Bell State")
    qc.h(0)
    qc.cnot(0, 1)
    qc.measure()
    return qc


def ghz_state(n: int = 3) -> QuantumCircuit:
    """
    GHZ (Greenberger–Horne–Zeilinger) state: (|000...0⟩ + |111...1⟩) / √2

    n-qubit maximally entangled state. Noise sensitive — deeper circuits
    with more CNOT gates accumulate more errors.

    Args:
        n: Number of qubits (default 3)
    """
    qc = QuantumCircuit(n, name=f"GHZ-{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cnot(i, i + 1)
    qc.measure()
    return qc


def qaoa_ansatz(layers: int = 2) -> QuantumCircuit:
    """
    QAOA-inspired variational ansatz circuit.

    Alternating cost (RZ) and mixer (RX) layers on 4 qubits,
    connected by CNOT entangling gates. Represents the kind of
    variational circuit common in near-term quantum optimization.

    Args:
        layers: Number of QAOA layers (default 2)
    """
    n_qubits = 4
    qc = QuantumCircuit(n_qubits, name=f"QAOA-{layers}L")

    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)

    # QAOA layers: cost + mixer
    gamma = math.pi / 4  # cost parameter
    beta = math.pi / 8   # mixer parameter

    for _ in range(layers):
        # Cost layer: RZ rotations + CNOT entanglement
        for q in range(n_qubits):
            qc.rz(q, gamma)
        for q in range(n_qubits - 1):
            qc.cnot(q, q + 1)

        # Mixer layer: RX rotations
        for q in range(n_qubits):
            qc.rx(q, 2 * beta)

    qc.measure()
    return qc


# Convenient list for demos
EXAMPLE_CIRCUITS = [
    bell_state(),
    ghz_state(3),
    qaoa_ansatz(2),
]
