"""
Quantum circuit linting and validation tools.

This module provides linting capabilities for quantum circuits, including
pulse-level analysis, gate constraint checking, and hardware compatibility
validation.
"""

import abc
import warnings
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import yaml

try:
    import qiskit
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False


@dataclass
class LintIssue:
    """Container for linting issues."""
    severity: str  # 'error', 'warning', 'info'
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None
    suggestion: Optional[str] = None
    time: Optional[float] = None  # For pulse-level issues


@dataclass
class LintResult:
    """Container for linting results."""
    issues: List[LintIssue]
    total_issues: int
    errors: int
    warnings: int
    infos: int
    
    def __post_init__(self):
        self.total_issues = len(self.issues)
        self.errors = sum(1 for issue in self.issues if issue.severity == 'error')
        self.warnings = sum(1 for issue in self.issues if issue.severity == 'warning')
        self.infos = sum(1 for issue in self.issues if issue.severity == 'info')
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.errors > 0
    
    def print_summary(self):
        """Print linting summary."""
        print(f"Linting completed: {self.total_issues} issues found")
        if self.errors > 0:
            print(f"  Errors: {self.errors}")
        if self.warnings > 0:
            print(f"  Warnings: {self.warnings}")
        if self.infos > 0:
            print(f"  Info: {self.infos}")


class QuantumLinter(abc.ABC):
    """
    Base class for quantum circuit linting.
    
    This class provides the foundation for linting quantum circuits
    across different frameworks and checking various constraints.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize quantum linter.
        
        Args:
            config_file: Path to linting configuration file
        """
        self.config = self._load_config(config_file)
        self.rules = self._initialize_rules()
    
    def lint_circuit(self, circuit, file_path: Optional[str] = None) -> LintResult:
        """
        Lint a quantum circuit.
        
        Args:
            circuit: Quantum circuit to lint
            file_path: Optional file path for context
            
        Returns:
            LintResult containing found issues
        """
        issues = []
        
        # Run all enabled rules
        for rule_name, rule_func in self.rules.items():
            if self._is_rule_enabled(rule_name):
                try:
                    rule_issues = rule_func(circuit, file_path)
                    issues.extend(rule_issues)
                except Exception as e:
                    # Add error for failed rule
                    issues.append(LintIssue(
                        severity='error',
                        message=f"Rule '{rule_name}' failed: {str(e)}",
                        file_path=file_path,
                        rule=rule_name
                    ))
        
        return LintResult(issues)
    
    def lint_directory(self, directory: str, pattern: str = "*.py") -> Dict[str, LintResult]:
        """
        Lint all quantum circuits in a directory.
        
        Args:
            directory: Directory path to scan
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping file paths to lint results
        """
        results = {}
        directory_path = Path(directory)
        
        for file_path in directory_path.glob(pattern):
            try:
                circuits = self._extract_circuits_from_file(file_path)
                for i, circuit in enumerate(circuits):
                    result = self.lint_circuit(circuit, str(file_path))
                    key = f"{file_path}#{i}" if len(circuits) > 1 else str(file_path)
                    results[key] = result
            except Exception as e:
                # Add file-level error
                results[str(file_path)] = LintResult([
                    LintIssue(
                        severity='error',
                        message=f"Failed to process file: {str(e)}",
                        file_path=str(file_path)
                    )
                ])
        
        return results
    
    @abc.abstractmethod
    def _initialize_rules(self) -> Dict[str, callable]:
        """Initialize linting rules."""
        pass
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load linting configuration."""
        default_config = {
            'gate_constraints': {
                'max_circuit_depth': 100,
                'max_two_qubit_gates': 50,
                'allowed_gates': []
            },
            'pulse_constraints': {
                'max_amplitude': 1.0,
                'min_pulse_duration': 16,
                'phase_granularity': 0.01
            },
            'rules': {
                'circuit-depth': True,
                'gate-compatibility': True,
                'pulse-constraints': True,
                'measurement-optimization': True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge with defaults
                self._deep_merge(default_config, user_config)
            except Exception as e:
                warnings.warn(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def _is_rule_enabled(self, rule_name: str) -> bool:
        """Check if a rule is enabled."""
        return self.config.get('rules', {}).get(rule_name, True)
    
    def _deep_merge(self, base: dict, update: dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _extract_circuits_from_file(self, file_path: Path) -> List:
        """Extract quantum circuits from Python file."""
        # Placeholder implementation
        # In reality, this would parse Python AST to find circuit definitions
        warnings.warn("Circuit extraction from files is not yet implemented")
        return []


class QiskitLinter(QuantumLinter):
    """Linter specifically for Qiskit circuits."""
    
    def _initialize_rules(self) -> Dict[str, callable]:
        """Initialize Qiskit-specific linting rules."""
        return {
            'circuit-depth': self._check_circuit_depth,
            'gate-compatibility': self._check_gate_compatibility,
            'measurement-optimization': self._check_measurement_optimization,
            'qubit-usage': self._check_qubit_usage,
            'classical-register': self._check_classical_register
        }
    
    def _check_circuit_depth(self, circuit, file_path: Optional[str] = None) -> List[LintIssue]:
        """Check circuit depth constraints."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return []
        
        issues = []
        max_depth = self.config['gate_constraints']['max_circuit_depth']
        
        circuit_depth = circuit.depth()
        if circuit_depth > max_depth:
            issues.append(LintIssue(
                severity='warning',
                message=f"Circuit depth {circuit_depth} exceeds maximum {max_depth}",
                file_path=file_path,
                rule='circuit-depth',
                suggestion=f"Consider optimizing circuit or splitting into smaller subcircuits"
            ))
        
        return issues
    
    def _check_gate_compatibility(self, circuit, file_path: Optional[str] = None) -> List[LintIssue]:
        """Check gate compatibility with target backend."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return []
        
        issues = []
        allowed_gates = self.config['gate_constraints'].get('allowed_gates', [])
        
        if allowed_gates:
            for instruction in circuit.data:
                gate_name = instruction.operation.name
                if gate_name not in allowed_gates:
                    issues.append(LintIssue(
                        severity='error',
                        message=f"Gate '{gate_name}' not allowed in target backend",
                        file_path=file_path,
                        rule='gate-compatibility',
                        suggestion=f"Replace with allowed gates: {', '.join(allowed_gates)}"
                    ))
        
        # Check two-qubit gate count
        max_two_qubit = self.config['gate_constraints']['max_two_qubit_gates']
        two_qubit_count = sum(1 for instr in circuit.data 
                             if len(instr.qubits) == 2)
        
        if two_qubit_count > max_two_qubit:
            issues.append(LintIssue(
                severity='warning',
                message=f"Two-qubit gate count {two_qubit_count} exceeds maximum {max_two_qubit}",
                file_path=file_path,
                rule='gate-compatibility',
                suggestion="Consider circuit optimization to reduce two-qubit gates"
            ))
        
        return issues
    
    def _check_measurement_optimization(self, circuit, file_path: Optional[str] = None) -> List[LintIssue]:
        """Check measurement placement and optimization."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return []
        
        issues = []
        
        # Check if measurements are at the end
        has_measurement = False
        measurement_indices = []
        
        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name == 'measure':
                has_measurement = True
                measurement_indices.append(i)
        
        if has_measurement:
            # Check if there are gates after measurements
            last_measurement = max(measurement_indices)
            if last_measurement < len(circuit.data) - 1:
                gates_after_measurement = len(circuit.data) - 1 - last_measurement
                issues.append(LintIssue(
                    severity='warning',
                    message=f"Found {gates_after_measurement} operations after measurement",
                    file_path=file_path,
                    rule='measurement-optimization',
                    suggestion="Move measurements to the end of the circuit"
                ))
        
        return issues
    
    def _check_qubit_usage(self, circuit, file_path: Optional[str] = None) -> List[LintIssue]:
        """Check qubit usage patterns."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return []
        
        issues = []
        
        # Check for unused qubits
        used_qubits = set()
        for instruction in circuit.data:
            for qubit in instruction.qubits:
                used_qubits.add(circuit.find_bit(qubit).index)
        
        total_qubits = circuit.num_qubits
        unused_qubits = set(range(total_qubits)) - used_qubits
        
        if unused_qubits:
            issues.append(LintIssue(
                severity='info',
                message=f"Unused qubits: {sorted(unused_qubits)}",
                file_path=file_path,
                rule='qubit-usage',
                suggestion="Consider removing unused qubits to optimize circuit"
            ))
        
        return issues
    
    def _check_classical_register(self, circuit, file_path: Optional[str] = None) -> List[LintIssue]:
        """Check classical register usage."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return []
        
        issues = []
        
        # Check if classical register size matches quantum register
        if circuit.num_clbits != circuit.num_qubits:
            issues.append(LintIssue(
                severity='info',
                message=f"Classical register size ({circuit.num_clbits}) differs from quantum register size ({circuit.num_qubits})",
                file_path=file_path,
                rule='classical-register',
                suggestion="Consider matching classical and quantum register sizes for full measurement"
            ))
        
        return issues


class PulseLinter:
    """
    Linter for pulse-level quantum programs.
    
    This linter checks pulse schedules for hardware constraint violations,
    amplitude limits, timing issues, and other pulse-specific problems.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize pulse linter.
        
        Args:
            config_file: Path to pulse linting configuration
        """
        self.config = self._load_pulse_config(config_file)
    
    @classmethod
    def from_config(cls, config_file: str) -> 'PulseLinter':
        """Create PulseLinter from configuration file."""
        return cls(config_file)
    
    def lint_schedule(self, pulse_schedule) -> List[LintIssue]:
        """
        Lint a pulse schedule.
        
        Args:
            pulse_schedule: Pulse schedule to analyze
            
        Returns:
            List of linting issues found
        """
        issues = []
        
        # Check amplitude constraints
        issues.extend(self._check_amplitude_constraints(pulse_schedule))
        
        # Check timing constraints
        issues.extend(self._check_timing_constraints(pulse_schedule))
        
        # Check frequency constraints
        issues.extend(self._check_frequency_constraints(pulse_schedule))
        
        # Check phase constraints
        issues.extend(self._check_phase_constraints(pulse_schedule))
        
        return issues
    
    def _load_pulse_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load pulse linting configuration."""
        default_config = {
            'pulse_constraints': {
                'max_amplitude': 1.0,
                'min_pulse_duration': 16,
                'phase_granularity': 0.01,
                'frequency_limits': []
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                warnings.warn(f"Failed to load pulse config {config_file}: {e}")
        
        return default_config
    
    def _check_amplitude_constraints(self, pulse_schedule) -> List[LintIssue]:
        """Check pulse amplitude constraints."""
        issues = []
        max_amplitude = self.config['pulse_constraints']['max_amplitude']
        
        # Placeholder implementation
        # Real implementation would analyze pulse schedule for amplitude violations
        
        return issues
    
    def _check_timing_constraints(self, pulse_schedule) -> List[LintIssue]:
        """Check pulse timing constraints."""
        issues = []
        min_duration = self.config['pulse_constraints']['min_pulse_duration']
        
        # Placeholder implementation
        # Real implementation would check pulse durations and timing overlaps
        
        return issues
    
    def _check_frequency_constraints(self, pulse_schedule) -> List[LintIssue]:
        """Check frequency constraints."""
        issues = []
        frequency_limits = self.config['pulse_constraints'].get('frequency_limits', [])
        
        # Placeholder implementation
        # Real implementation would validate frequencies against hardware limits
        
        return issues
    
    def _check_phase_constraints(self, pulse_schedule) -> List[LintIssue]:
        """Check phase granularity constraints."""
        issues = []
        phase_granularity = self.config['pulse_constraints']['phase_granularity']
        
        # Placeholder implementation
        # Real implementation would check phase precision
        
        return issues


# CLI interface for linting
class LintingCLI:
    """Command-line interface for quantum linting."""
    
    def __init__(self):
        self.linters = {
            'qiskit': QiskitLinter,
            'pulse': PulseLinter
        }
    
    def run(self, framework: str, target: str, config: Optional[str] = None) -> int:
        """
        Run linting from command line.
        
        Args:
            framework: Framework to use ('qiskit', 'pulse')
            target: File or directory to lint
            config: Configuration file path
            
        Returns:
            Exit code (0 for success, 1 for errors)
        """
        try:
            if framework not in self.linters:
                print(f"Error: Unknown framework '{framework}'")
                return 1
            
            linter_class = self.linters[framework]
            linter = linter_class(config)
            
            if Path(target).is_file():
                # Lint single file
                results = {target: self._lint_file(linter, target)}
            else:
                # Lint directory
                results = linter.lint_directory(target)
            
            # Print results
            total_errors = 0
            for file_path, result in results.items():
                if result.total_issues > 0:
                    print(f"\n{file_path}:")
                    for issue in result.issues:
                        print(f"  {issue.severity.upper()}: {issue.message}")
                        if issue.suggestion:
                            print(f"    Suggestion: {issue.suggestion}")
                
                total_errors += result.errors
            
            # Print summary
            print(f"\nLinting completed. Total files: {len(results)}")
            print(f"Total errors: {total_errors}")
            
            return 1 if total_errors > 0 else 0
            
        except Exception as e:
            print(f"Error during linting: {e}")
            return 1
    
    def _lint_file(self, linter, file_path: str) -> LintResult:
        """Lint a single file."""
        # This would extract circuits from file and lint them
        # For now, return empty result
        return LintResult([])


def main():
    """Main entry point for quantum-lint CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum circuit linter')
    parser.add_argument('target', help='File or directory to lint')
    parser.add_argument('--framework', choices=['qiskit', 'pulse'], default='qiskit',
                       help='Quantum framework to use')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    
    args = parser.parse_args()
    
    cli = LintingCLI()
    exit_code = cli.run(args.framework, args.target, args.config)
    exit(exit_code)


if __name__ == '__main__':
    main()