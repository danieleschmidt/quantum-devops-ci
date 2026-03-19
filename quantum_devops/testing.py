"""
NoiseAwareTestRunner: CI-style test runner for quantum circuits.

Runs each circuit at multiple noise levels, checks fidelity thresholds,
and produces structured reports suitable for CI/CD pipelines.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .circuit import QuantumCircuit
from .simulator import NoisySimulator


@dataclass
class TestResult:
    """Result of testing a circuit at a single noise level."""
    circuit_name: str
    noise_level: float
    fidelity: float
    threshold: float
    shots: int
    counts: Dict[str, int]
    passed: bool
    duration_ms: float
    validation_issues: List[str]

    @property
    def status(self) -> str:
        if self.validation_issues:
            return "INVALID"
        return "PASS" if self.passed else "FAIL"

    def summary(self) -> str:
        issues = f" [{'; '.join(self.validation_issues)}]" if self.validation_issues else ""
        return (
            f"[{self.status}] {self.circuit_name} @ noise={self.noise_level:.3f} "
            f"fidelity={self.fidelity:.4f} (threshold={self.threshold:.4f}) "
            f"{self.shots} shots, {self.duration_ms:.1f}ms{issues}"
        )


@dataclass
class CircuitTestReport:
    """Aggregated test report for one circuit across all noise levels."""
    circuit_name: str
    num_qubits: int
    gate_count: int
    depth: int
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def max_tolerable_noise(self) -> Optional[float]:
        """Highest noise level at which circuit still passes fidelity threshold."""
        passing = [r.noise_level for r in self.results if r.passed]
        return max(passing) if passing else None

    @property
    def noise_sensitivity(self) -> str:
        """Qualitative noise sensitivity classification."""
        max_noise = self.max_tolerable_noise
        if max_noise is None:
            return "CRITICAL"
        if max_noise >= 0.05:
            return "ROBUST"
        if max_noise >= 0.01:
            return "MODERATE"
        return "SENSITIVE"

    def report(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Circuit: {self.circuit_name}",
            f"  Qubits: {self.num_qubits}  Depth: {self.depth}  Gates: {self.gate_count}",
            f"  Overall: {'PASS' if self.passed else 'FAIL'}  "
            f"Noise sensitivity: {self.noise_sensitivity}",
            f"  Max tolerable noise: {self.max_tolerable_noise}",
            f"{'─'*60}",
        ]
        for result in self.results:
            lines.append(f"  {result.summary()}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


class NoiseAwareTestRunner:
    """
    Runs quantum circuits at multiple noise levels and checks fidelity.

    Args:
        noise_levels: List of depolarizing error rates to test at
        shots: Number of simulation shots per noise level
        fidelity_threshold: Minimum acceptable fidelity (0.0–1.0)
        seed: Optional RNG seed for reproducibility
    """

    DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

    def __init__(
        self,
        noise_levels: Optional[List[float]] = None,
        shots: int = 200,
        fidelity_threshold: float = 0.90,
        seed: Optional[int] = None,
    ):
        self.noise_levels = noise_levels if noise_levels is not None else self.DEFAULT_NOISE_LEVELS
        self.shots = shots
        self.fidelity_threshold = fidelity_threshold
        self.seed = seed

    def test_circuit(self, circuit: QuantumCircuit) -> CircuitTestReport:
        """
        Run a single circuit across all configured noise levels.

        Returns:
            CircuitTestReport with per-noise-level results.
        """
        validation_issues = circuit.validate()
        report = CircuitTestReport(
            circuit_name=circuit.name,
            num_qubits=circuit.num_qubits,
            gate_count=circuit.gate_count,
            depth=circuit.depth,
        )

        for noise in self.noise_levels:
            sim = NoisySimulator(
                error_rate=noise,
                seed=self.seed,
            )
            t0 = time.monotonic()
            stats = sim.run_and_aggregate(circuit, self.shots)
            elapsed_ms = (time.monotonic() - t0) * 1000

            passed = (
                stats["fidelity"] >= self.fidelity_threshold
                and not validation_issues
            )

            result = TestResult(
                circuit_name=circuit.name,
                noise_level=noise,
                fidelity=stats["fidelity"],
                threshold=self.fidelity_threshold,
                shots=self.shots,
                counts=stats["counts"],
                passed=passed,
                duration_ms=elapsed_ms,
                validation_issues=validation_issues,
            )
            report.results.append(result)

        return report

    def test_suite(
        self, circuits: List[QuantumCircuit]
    ) -> Tuple[List[CircuitTestReport], bool]:
        """
        Run a suite of circuits. Returns (reports, all_passed).

        Args:
            circuits: List of QuantumCircuit instances to test

        Returns:
            Tuple of (list of CircuitTestReport, bool all_passed)
        """
        reports = [self.test_circuit(c) for c in circuits]
        all_passed = all(r.passed for r in reports)
        return reports, all_passed

    def print_suite_summary(
        self,
        reports: List[CircuitTestReport],
        all_passed: bool,
    ) -> None:
        """Print a CI-friendly test summary to stdout."""
        print("\n" + "=" * 60)
        print("QUANTUM CI TEST SUITE RESULTS")
        print("=" * 60)
        for report in reports:
            print(report.report())

        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        print(f"\nSummary: {passed}/{total} circuits passed")
        print(f"Overall: {'✅ PASS' if all_passed else '❌ FAIL'}")
        print("=" * 60 + "\n")
