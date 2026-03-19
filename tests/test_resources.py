"""Tests for ResourceEstimator."""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_devops.circuit import QuantumCircuit
from quantum_devops.resources import ResourceEstimator, ResourceReport, _is_t_gate
from quantum_devops.circuit import Gate


class TestIsTGate:
    def test_rz_pi_over_4_is_t(self):
        g = Gate("RZ", (0,), (math.pi / 4,))
        assert _is_t_gate(g)

    def test_rz_pi_is_not_t(self):
        g = Gate("RZ", (0,), (math.pi,))
        assert not _is_t_gate(g)

    def test_rz_zero_is_not_t(self):
        g = Gate("RZ", (0,), (0.0,))
        assert not _is_t_gate(g)

    def test_h_is_not_t(self):
        g = Gate("H", (0,))
        assert not _is_t_gate(g)


class TestResourceReport:
    def _make_report(self, **kwargs):
        defaults = dict(
            circuit_name="test", qubit_count=2, depth=3,
            gate_count=3, two_qubit_gate_count=1, t_gate_count=0,
            clifford_gate_count=2, measurement_count=2,
            estimated_runtime_us=0.5,
        )
        defaults.update(kwargs)
        return ResourceReport(**defaults)

    def test_zero_t_clifford_only(self):
        r = self._make_report(t_gate_count=0)
        assert "Clifford-only" in r.fault_tolerance_overhead

    def test_low_t_count(self):
        r = self._make_report(t_gate_count=10)
        assert "Low" in r.fault_tolerance_overhead

    def test_high_t_count(self):
        r = self._make_report(t_gate_count=600)
        assert "High" in r.fault_tolerance_overhead

    def test_summary_contains_fields(self):
        r = self._make_report()
        s = r.summary()
        assert "Qubits" in s
        assert "Circuit depth" in s
        assert "T-gates" in s

    def test_to_dict(self):
        r = self._make_report()
        d = r.to_dict()
        assert "qubit_count" in d
        assert "t_gate_count" in d
        assert "name" in d


class TestResourceEstimator:
    def test_bell_state_resources(self):
        qc = QuantumCircuit(2, name="Bell")
        qc.h(0).cnot(0, 1).measure()

        est = ResourceEstimator()
        report = est.estimate(qc)

        assert report.qubit_count == 2
        assert report.gate_count == 2  # H + CNOT (no measurements)
        assert report.two_qubit_gate_count == 1  # CNOT
        assert report.t_gate_count == 0  # No T gates
        assert report.measurement_count == 2

    def test_rz_pi_over_4_counted_as_t(self):
        qc = QuantumCircuit(1)
        qc.rz(0, math.pi / 4)
        qc.measure()

        est = ResourceEstimator()
        report = est.estimate(qc)
        assert report.t_gate_count == 1

    def test_multiple_circuits(self):
        est = ResourceEstimator()
        circuits = [
            QuantumCircuit(1).h(0).measure(),
            QuantumCircuit(2).h(0).cnot(0, 1).measure(),
        ]
        reports = est.estimate_suite(circuits)
        assert len(reports) == 2

    def test_depth_warning(self):
        qc = QuantumCircuit(1, name="Deep")
        for _ in range(110):
            qc.h(0)
        qc.measure()

        est = ResourceEstimator()
        report = est.estimate(qc)
        assert any("depth" in w.lower() for w in report.warnings)

    def test_budget_check_pass(self):
        qc = QuantumCircuit(2, name="Bell")
        qc.h(0).cnot(0, 1).measure()

        est = ResourceEstimator()
        report, violations = est.budget_check(
            qc, max_depth=10, max_t_gates=10, max_two_qubit_gates=5
        )
        assert violations == []

    def test_budget_check_fail_depth(self):
        qc = QuantumCircuit(1, name="Deep")
        for _ in range(5):
            qc.h(0)
        qc.measure()

        est = ResourceEstimator()
        _, violations = est.budget_check(qc, max_depth=2)
        assert any("Depth" in v for v in violations)

    def test_budget_check_fail_t_gates(self):
        qc = QuantumCircuit(1, name="TGatey")
        for _ in range(10):
            qc.rz(0, math.pi / 4)
        qc.measure()

        est = ResourceEstimator()
        _, violations = est.budget_check(qc, max_t_gates=5)
        assert any("T-gate" in v for v in violations)

    def test_gate_breakdown(self):
        qc = QuantumCircuit(2)
        qc.h(0).h(1).cnot(0, 1).measure()

        est = ResourceEstimator()
        report = est.estimate(qc)
        assert report.gate_breakdown.get("H", 0) == 2
        assert report.gate_breakdown.get("CNOT", 0) == 1

    def test_runtime_nonzero(self):
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure()

        est = ResourceEstimator()
        report = est.estimate(qc)
        assert report.estimated_runtime_us > 0
