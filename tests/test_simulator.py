"""Tests for NoisySimulator."""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_devops.circuit import QuantumCircuit
from quantum_devops.simulator import NoisySimulator, QubitState


class TestQubitState:
    def test_initial_state(self):
        q = QubitState()
        assert abs(q.prob_zero - 1.0) < 1e-10
        assert abs(q.prob_one - 0.0) < 1e-10

    def test_hadamard_creates_superposition(self):
        q = QubitState()
        q.apply_h()
        assert abs(q.prob_zero - 0.5) < 1e-10
        assert abs(q.prob_one - 0.5) < 1e-10

    def test_hadamard_twice_is_identity(self):
        q = QubitState()
        q.apply_h()
        q.apply_h()
        assert abs(q.prob_zero - 1.0) < 1e-6

    def test_pauli_x_flips(self):
        q = QubitState()
        q.apply_pauli_x()
        assert abs(q.prob_one - 1.0) < 1e-10

    def test_rz_phase(self):
        q = QubitState()
        q.apply_h()
        p0_before = q.prob_zero
        q.apply_rz(math.pi / 4)
        # RZ changes phase but not populations in |+⟩
        assert abs(q.prob_zero - p0_before) < 1e-10

    def test_rx_rotation(self):
        q = QubitState()
        q.apply_rx(math.pi)
        # RX(π)|0⟩ = |1⟩ (bit flip)
        assert abs(q.prob_one - 1.0) < 1e-10

    def test_normalization(self):
        q = QubitState()
        q.apply_h()
        assert abs(q.prob_zero + q.prob_one - 1.0) < 1e-10

    def test_measurement_collapses(self):
        import random
        rng = random.Random(42)
        q = QubitState()
        outcome = q.measure(rng)
        assert outcome in (0, 1)
        # After measurement, state is definite
        if outcome == 0:
            assert abs(q.prob_zero - 1.0) < 1e-10
        else:
            assert abs(q.prob_one - 1.0) < 1e-10


class TestNoisySimulator:
    def test_invalid_error_rate_raises(self):
        with pytest.raises(ValueError, match="error_rate"):
            NoisySimulator(error_rate=-0.1)

        with pytest.raises(ValueError, match="error_rate"):
            NoisySimulator(error_rate=1.5)

    def test_zero_noise_bell_state(self):
        """Bell state: simulator produces measurement outcomes for all qubits."""
        qc = QuantumCircuit(2, name="Bell")
        qc.h(0).cnot(0, 1).measure()

        sim = NoisySimulator(error_rate=0.0, seed=42)
        stats = sim.run_and_aggregate(qc, shots=500)

        counts = stats["counts"]
        total = sum(counts.values())
        assert total == 500

        # All outcomes should be 2-bit strings
        for key in counts:
            assert len(key) == 2 and all(c in "01" for c in key)

        # At zero noise, fidelity vs ideal should be 1.0 (same RNG, same runs)
        assert stats["fidelity"] >= 0.95

    def test_zero_noise_single_qubit(self):
        """H+measure on single qubit: should be ~50/50."""
        qc = QuantumCircuit(1)
        qc.h(0).measure()

        sim = NoisySimulator(error_rate=0.0, seed=0)
        stats = sim.run_and_aggregate(qc, shots=1000)
        counts = stats["counts"]

        zero_frac = counts.get("0", 0) / 1000
        one_frac = counts.get("1", 0) / 1000
        assert abs(zero_frac - 0.5) < 0.1
        assert abs(one_frac - 0.5) < 0.1

    def test_fidelity_perfect_noise_zero(self):
        """Zero noise should give fidelity ≈ 1."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure()

        sim = NoisySimulator(error_rate=0.0, seed=42)
        stats = sim.run_and_aggregate(qc, shots=200)
        assert stats["fidelity"] > 0.95

    def test_high_noise_reduces_fidelity(self):
        """High noise should reduce fidelity compared to zero noise."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure()

        sim_clean = NoisySimulator(error_rate=0.0, seed=42)
        sim_noisy = NoisySimulator(error_rate=0.5, seed=42)

        stats_clean = sim_clean.run_and_aggregate(qc, shots=200)
        stats_noisy = sim_noisy.run_and_aggregate(qc, shots=200)

        assert stats_clean["fidelity"] >= stats_noisy["fidelity"]

    def test_run_returns_measurements(self):
        qc = QuantumCircuit(2)
        qc.h(0).measure()

        sim = NoisySimulator(error_rate=0.0, seed=0)
        results = sim.run(qc, shots=5)
        assert len(results) == 5
        for r in results:
            assert 0 in r
            assert r[0] in (0, 1)

    def test_aggregate_returns_expected_keys(self):
        qc = QuantumCircuit(1)
        qc.h(0).measure()

        sim = NoisySimulator(error_rate=0.0, seed=1)
        stats = sim.run_and_aggregate(qc, shots=100)

        assert "counts" in stats
        assert "fidelity" in stats
        assert "error_rate" in stats
        assert "shots" in stats
        assert stats["shots"] == 100
        assert stats["error_rate"] == 0.0

    def test_circuit_without_measurements(self):
        """Circuit with no measurements should return empty dicts."""
        qc = QuantumCircuit(1)
        qc.h(0)  # No measure

        sim = NoisySimulator(error_rate=0.0, seed=0)
        results = sim.run(qc, shots=3)
        for r in results:
            assert r == {}

    def test_seed_reproducibility(self):
        """Same seed should give same results."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure()

        sim1 = NoisySimulator(error_rate=0.05, seed=99)
        sim2 = NoisySimulator(error_rate=0.05, seed=99)

        stats1 = sim1.run_and_aggregate(qc, shots=100)
        stats2 = sim2.run_and_aggregate(qc, shots=100)

        assert stats1["counts"] == stats2["counts"]
        assert stats1["fidelity"] == stats2["fidelity"]
