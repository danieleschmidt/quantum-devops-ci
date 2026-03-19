"""Tests for NoiseAwareTestRunner."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_devops.circuit import QuantumCircuit
from quantum_devops.testing import NoiseAwareTestRunner, TestResult, CircuitTestReport


def make_bell():
    qc = QuantumCircuit(2, name="Bell")
    qc.h(0).cnot(0, 1).measure()
    return qc


def make_invalid_circuit():
    """Circuit with no measurements — should fail validation."""
    qc = QuantumCircuit(1, name="InvalidNoMeasure")
    qc.h(0)
    return qc


class TestTestResult:
    def test_pass_status(self):
        r = TestResult(
            circuit_name="Bell", noise_level=0.0, fidelity=0.99,
            threshold=0.85, shots=100, counts={"00": 50, "11": 50},
            passed=True, duration_ms=5.0, validation_issues=[]
        )
        assert r.status == "PASS"

    def test_fail_status(self):
        r = TestResult(
            circuit_name="Bell", noise_level=0.5, fidelity=0.30,
            threshold=0.85, shots=100, counts={},
            passed=False, duration_ms=5.0, validation_issues=[]
        )
        assert r.status == "FAIL"

    def test_invalid_status(self):
        r = TestResult(
            circuit_name="Bad", noise_level=0.0, fidelity=0.0,
            threshold=0.85, shots=100, counts={},
            passed=False, duration_ms=1.0, validation_issues=["No measurements"]
        )
        assert r.status == "INVALID"

    def test_summary_contains_info(self):
        r = TestResult(
            circuit_name="Bell", noise_level=0.01, fidelity=0.95,
            threshold=0.85, shots=100, counts={"00": 50, "11": 50},
            passed=True, duration_ms=8.5, validation_issues=[]
        )
        s = r.summary()
        assert "Bell" in s
        assert "0.010" in s
        assert "PASS" in s


class TestCircuitTestReport:
    def test_all_pass(self):
        report = CircuitTestReport("Bell", 2, 2, 2)
        for noise in [0.0, 0.01]:
            report.results.append(TestResult(
                circuit_name="Bell", noise_level=noise, fidelity=0.99,
                threshold=0.85, shots=100, counts={},
                passed=True, duration_ms=5.0, validation_issues=[]
            ))
        assert report.passed

    def test_one_fail(self):
        report = CircuitTestReport("Bell", 2, 2, 2)
        report.results.append(TestResult(
            circuit_name="Bell", noise_level=0.1, fidelity=0.5,
            threshold=0.85, shots=100, counts={},
            passed=False, duration_ms=5.0, validation_issues=[]
        ))
        assert not report.passed

    def test_max_tolerable_noise(self):
        report = CircuitTestReport("Bell", 2, 2, 2)
        for noise, passed in [(0.0, True), (0.01, True), (0.1, False)]:
            report.results.append(TestResult(
                circuit_name="Bell", noise_level=noise, fidelity=0.9,
                threshold=0.85, shots=100, counts={},
                passed=passed, duration_ms=5.0, validation_issues=[]
            ))
        assert report.max_tolerable_noise == 0.01

    def test_noise_sensitivity_robust(self):
        report = CircuitTestReport("Bell", 2, 2, 2)
        report.results.append(TestResult(
            "Bell", 0.1, 0.99, 0.85, 100, {}, True, 5.0, []
        ))
        assert report.noise_sensitivity == "ROBUST"

    def test_noise_sensitivity_critical(self):
        report = CircuitTestReport("Fragile", 2, 2, 2)
        report.results.append(TestResult(
            "Fragile", 0.001, 0.5, 0.85, 100, {}, False, 5.0, []
        ))
        assert report.noise_sensitivity == "CRITICAL"


class TestNoiseAwareTestRunner:
    def test_test_circuit_returns_report(self):
        runner = NoiseAwareTestRunner(
            noise_levels=[0.0, 0.01],
            shots=50,
            fidelity_threshold=0.7,
            seed=42,
        )
        qc = make_bell()
        report = runner.test_circuit(qc)

        assert isinstance(report, CircuitTestReport)
        assert report.circuit_name == "Bell"
        assert len(report.results) == 2

    def test_zero_noise_passes_fidelity(self):
        runner = NoiseAwareTestRunner(
            noise_levels=[0.0],
            shots=100,
            fidelity_threshold=0.9,
            seed=42,
        )
        qc = make_bell()
        report = runner.test_circuit(qc)
        # Zero noise should produce high fidelity
        assert report.results[0].fidelity > 0.9

    def test_invalid_circuit_fails(self):
        runner = NoiseAwareTestRunner(
            noise_levels=[0.0],
            shots=50,
            fidelity_threshold=0.85,
            seed=42,
        )
        qc = make_invalid_circuit()
        report = runner.test_circuit(qc)
        # Validation issues should cause failure
        for result in report.results:
            assert not result.passed
            assert result.validation_issues

    def test_suite_all_pass(self):
        runner = NoiseAwareTestRunner(
            noise_levels=[0.0],
            shots=50,
            fidelity_threshold=0.7,
            seed=42,
        )
        circuits = [make_bell()]
        reports, all_passed = runner.test_suite(circuits)
        assert len(reports) == 1
        assert all_passed  # Zero noise with low threshold should pass

    def test_suite_returns_correct_count(self):
        runner = NoiseAwareTestRunner(
            noise_levels=[0.0, 0.05],
            shots=50,
            seed=42,
        )
        circuits = [make_bell(), make_bell()]
        reports, _ = runner.test_suite(circuits)
        assert len(reports) == 2
        for r in reports:
            assert len(r.results) == 2

    def test_high_threshold_causes_fail_at_very_high_noise(self):
        """Unreachable fidelity threshold should always fail."""
        runner = NoiseAwareTestRunner(
            noise_levels=[0.5],
            shots=100,
            fidelity_threshold=1.01,  # Impossible threshold > 1.0
            seed=42,
        )
        qc = make_bell()
        report = runner.test_circuit(qc)
        # fidelity is always <= 1.0 so this must fail
        assert not report.passed
