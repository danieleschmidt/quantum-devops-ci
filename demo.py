#!/usr/bin/env python3
"""
quantum-devops-ci demo: Test 3 example circuits at noise levels 0, 0.01, 0.1

Demonstrates the full toolkit:
  - Circuit construction and validation
  - Noisy simulation with depolarizing errors
  - CI-style test runner with fidelity thresholds
  - Resource estimation and T-gate budgets
  - GitHub Actions YAML generation
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from quantum_devops import (
    NoiseAwareTestRunner,
    CITemplate,
    ResourceEstimator,
)
from examples.circuits import bell_state, ghz_state, qaoa_ansatz


def main():
    print("\n" + "█" * 60)
    print("  quantum-devops-ci  |  Noise-Aware Quantum CI/CD Toolkit")
    print("█" * 60)

    # ── 1. Build example circuits ────────────────────────────────────────────
    circuits = [
        bell_state(),
        ghz_state(3),
        qaoa_ansatz(2),
    ]

    print("\n📋 Circuits to test:")
    for c in circuits:
        issues = c.validate()
        issue_str = f"  ⚠️  {', '.join(issues)}" if issues else "  ✓ valid"
        print(f"   {c}{issue_str}")

    # ── 2. Noise-aware test run ──────────────────────────────────────────────
    print("\n🔬 Running noise-aware tests at levels: 0, 0.01, 0.1 ...")
    runner = NoiseAwareTestRunner(
        noise_levels=[0.0, 0.01, 0.1],
        shots=300,
        fidelity_threshold=0.85,
        seed=42,
    )

    reports, all_passed = runner.test_suite(circuits)
    runner.print_suite_summary(reports, all_passed)

    # ── 3. Resource estimation ───────────────────────────────────────────────
    print("\n⚙️  Resource Estimation:")
    print("─" * 60)
    estimator = ResourceEstimator()
    for circuit in circuits:
        report = estimator.estimate(circuit)
        print(report.summary())
        print()

    # Budget check example
    print("🔍 Budget check (max depth=20, max T-gates=50):")
    for circuit in circuits:
        _, violations = estimator.budget_check(
            circuit, max_depth=20, max_t_gates=50
        )
        status = "✅ within budget" if not violations else "❌ " + "; ".join(violations)
        print(f"   {circuit.name}: {status}")

    # ── 4. CI Template generation ────────────────────────────────────────────
    print("\n📄 Generated GitHub Actions workflow (preview):")
    print("─" * 60)
    template = CITemplate(
        repo_name="quantum-devops-ci",
        python_versions=["3.11", "3.12"],
        noise_levels=[0.0, 0.01, 0.05, 0.1],
        fidelity_threshold=0.85,
        shots=300,
    )
    yaml = template.generate()
    # Print just the first 30 lines
    preview_lines = yaml.split("\n")[:30]
    print("\n".join(preview_lines))
    print("  ... (truncated — run template.save() to write full file)")

    # Save the actual workflow
    workflow_path = os.path.join(os.path.dirname(__file__), ".github", "workflows", "quantum-ci.yml")
    template.save(workflow_path)
    print(f"\n✅ Workflow saved to: {workflow_path}")

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    exit_code = 0 if all_passed else 1
    print(f"  Final result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("█" * 60 + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
