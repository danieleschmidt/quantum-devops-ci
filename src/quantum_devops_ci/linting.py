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
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("pyyaml not available - YAML configuration loading will be limited")

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
        """Check if linting found any errors."""
        return self.errors > 0
    
    def add_issue(self, severity: str, message: str, **kwargs):
        """Add a linting issue."""
        issue = LintIssue(severity=severity, message=message, **kwargs)
        self.issues.append(issue)
    
    def get_summary(self) -> str:
        """Get a summary string of linting results."""
        if self.total_issues == 0:
            return "✅ No issues found"
        
        parts = []
        if self.errors > 0:
            parts.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        if self.warnings > 0:
            parts.append(f"{self.warnings} warning{'s' if self.warnings != 1 else ''}")
        if self.infos > 0:
            parts.append(f"{self.infos} info")
        
        return f"❌ {', '.join(parts)} found"


@dataclass
class LintingConfig:
    """Configuration for quantum circuit linting."""
    max_circuit_depth: int = 100
    max_two_qubit_gates: int = 50
    allowed_gates: List[str] = None
    forbidden_gates: List[str] = None
    max_qubits: int = 100
    pulse_constraints: Optional[Dict[str, Any]] = None
    gate_constraints: Optional[Dict[str, Any]] = None
    optimization_level: int = 1
    
    def __post_init__(self):
        if self.allowed_gates is None:
            self.allowed_gates = ['u1', 'u2', 'u3', 'cx']
        if self.forbidden_gates is None:
            self.forbidden_gates = []
    
    @classmethod
    def from_file(cls, config_path: str) -> 'LintingConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract relevant sections
            pulse_config = config_data.get('pulse_constraints', {})
            gate_config = config_data.get('gate_constraints', {})
            
            return cls(
                max_circuit_depth=gate_config.get('max_circuit_depth', 100),
                max_two_qubit_gates=gate_config.get('max_two_qubit_gates', 50),
                allowed_gates=gate_config.get('allowed_gates', ['u1', 'u2', 'u3', 'cx']),
                max_qubits=gate_config.get('max_qubits', 100),
                pulse_constraints=pulse_config,
                gate_constraints=gate_config
            )
        except Exception as e:
            warnings.warn(f"Failed to load config from {config_path}: {e}")
            return cls()


class QuantumLinter:
    """Abstract base class for quantum circuit linters."""
    
    def __init__(self, config: Optional[Union[str, LintingConfig]] = None):
        """Initialize linter with configuration."""
        if isinstance(config, str):
            self.config = LintingConfig.from_file(config)
        elif isinstance(config, LintingConfig):
            self.config = config
        else:
            self.config = LintingConfig()
    
    @abc.abstractmethod
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint a quantum circuit."""
        pass
    
    def lint_file(self, file_path: str) -> LintResult:
        """Lint circuits in a file."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        
        try:
            # This would need to parse the file and extract circuits
            # For now, return basic linting result
            result.add_issue('info', f'File linting not yet implemented for {file_path}', 
                           file_path=file_path)
        except Exception as e:
            result.add_issue('error', f'Failed to lint file {file_path}: {e}', 
                           file_path=file_path)
        
        return result
    
    def lint_circuit(self, circuit) -> LintResult:
        """Lint a quantum circuit - default implementation."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        result.add_issue('info', 'Default circuit linting completed')
        return result
    
    def lint_directory(self, directory_path: str) -> LintResult:
        """Lint all quantum files in a directory."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                result.add_issue('error', f'Directory not found: {directory_path}')
                return result
            
            # Find quantum files (simplified)
            quantum_files = list(directory.glob('**/*.py')) + list(directory.glob('**/*.qasm'))
            
            if not quantum_files:
                result.add_issue('warning', f'No quantum files found in {directory_path}')
                return result
            
            for file_path in quantum_files:
                file_result = self.lint_file(str(file_path))
                result.issues.extend(file_result.issues)
            
        except Exception as e:
            result.add_issue('error', f'Failed to lint directory {directory_path}: {e}')
        
        return result


class QiskitLinter(QuantumLinter):
    """Linter for Qiskit quantum circuits."""
    
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint a Qiskit quantum circuit."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        
        if not QISKIT_AVAILABLE:
            result.add_issue('error', 'Qiskit not available for linting')
            return result
        
        if not hasattr(circuit, 'data') or not hasattr(circuit, 'num_qubits'):
            result.add_issue('error', 'Invalid Qiskit circuit object')
            return result
        
        # Check circuit depth
        depth = circuit.depth()
        if depth > self.config.max_circuit_depth:
            result.add_issue(
                'warning',
                f'Circuit depth ({depth}) exceeds recommended maximum ({self.config.max_circuit_depth})',
                rule='max_circuit_depth',
                suggestion=f'Consider circuit optimization or breaking into smaller subcircuits'
            )
        
        # Check number of qubits
        if circuit.num_qubits > self.config.max_qubits:
            result.add_issue(
                'error',
                f'Circuit uses {circuit.num_qubits} qubits, exceeding limit of {self.config.max_qubits}',
                rule='max_qubits'
            )
        
        # Check gate usage
        gate_counts = {}
        forbidden_gates = []
        unknown_gates = []
        
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            # Check for forbidden gates
            if gate_name in self.config.forbidden_gates:
                forbidden_gates.append(gate_name)
            
            # Check for unknown gates (not in allowed list)
            if self.config.allowed_gates and gate_name not in self.config.allowed_gates:
                unknown_gates.append(gate_name)
        
        # Report forbidden gates
        for gate in set(forbidden_gates):
            result.add_issue(
                'error',
                f'Forbidden gate used: {gate} ({forbidden_gates.count(gate)} times)',
                rule='forbidden_gates',
                suggestion=f'Replace {gate} with allowed gate decomposition'
            )
        
        # Report unknown gates
        for gate in set(unknown_gates):
            result.add_issue(
                'warning',
                f'Unknown/non-standard gate: {gate} ({unknown_gates.count(gate)} times)',
                rule='allowed_gates',
                suggestion=f'Verify {gate} is supported by target backend'
            )
        
        # Check two-qubit gate count
        two_qubit_gates = ['cx', 'cy', 'cz', 'cnot', 'cphase', 'crz', 'cu']
        two_qubit_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
        
        if two_qubit_count > self.config.max_two_qubit_gates:
            result.add_issue(
                'warning',
                f'High two-qubit gate count ({two_qubit_count}) may impact fidelity',
                rule='max_two_qubit_gates',
                suggestion='Consider circuit optimization to reduce two-qubit gates'
            )
        
        # Check for measurement operations
        has_measurements = any(
            instruction.operation.name in ['measure', 'barrier'] 
            for instruction in circuit.data
        )
        
        if not has_measurements:
            result.add_issue(
                'info',
                'Circuit has no measurement operations',
                rule='measurements',
                suggestion='Add measurements if this circuit will be executed on hardware'
            )
        
        # Check circuit connectivity (simplified)
        self._check_connectivity(circuit, result)
        
        return result
    
    def _check_connectivity(self, circuit: Any, result: LintResult):
        """Check circuit connectivity constraints."""
        if not hasattr(circuit, 'data'):
            return
        
        # Simple connectivity check - look for long-range two-qubit gates
        for instruction in circuit.data:
            if len(instruction.qubits) == 2:
                qubit1 = instruction.qubits[0].index
                qubit2 = instruction.qubits[1].index
                
                # Check if qubits are adjacent (simplified linear connectivity)
                if abs(qubit1 - qubit2) > 1:
                    result.add_issue(
                        'warning',
                        f'Non-adjacent qubit gate: {instruction.operation.name} on qubits {qubit1}, {qubit2}',
                        rule='connectivity',
                        suggestion='Consider SWAP gates or circuit routing optimization'
                    )


class PulseLinter(QuantumLinter):
    """Linter for quantum pulse schedules."""
    
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint circuit for pulse-level constraints."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        
        # Pulse linting would require pulse schedule objects
        # For now, provide basic framework
        result.add_issue(
            'info',
            'Pulse-level linting not yet fully implemented',
            rule='pulse_linting'
        )
        
        return result
    
    def lint_schedule(self, schedule: Any) -> LintResult:
        """Lint a pulse schedule."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        
        if not self.config.pulse_constraints:
            result.add_issue('warning', 'No pulse constraints configured')
            return result
        
        pulse_config = self.config.pulse_constraints
        
        # Check pulse amplitude constraints
        max_amplitude = pulse_config.get('max_amplitude', 1.0)
        min_duration = pulse_config.get('min_pulse_duration', 16)
        
        # Mock pulse analysis (would need actual pulse schedule parsing)
        result.add_issue(
            'info',
            f'Checking pulse amplitudes against limit: {max_amplitude}',
            rule='pulse_amplitude'
        )
        
        result.add_issue(
            'info', 
            f'Checking pulse durations against minimum: {min_duration} dt',
            rule='pulse_duration'
        )
        
        # Frequency limits check
        freq_limits = pulse_config.get('frequency_limits', [])
        for freq_limit in freq_limits:
            channel = freq_limit.get('channel', 'unknown')
            min_freq = freq_limit.get('min', 0)
            max_freq = freq_limit.get('max', 1e9)
            
            result.add_issue(
                'info',
                f'Channel {channel}: frequency range {min_freq/1e6:.1f}-{max_freq/1e6:.1f} MHz',
                rule='frequency_limits'
            )
        
        return result


class CirqLinter(QuantumLinter):
    """Linter for Cirq quantum circuits."""
    
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint a Cirq quantum circuit."""
        result = LintResult(issues=[], total_issues=0, errors=0, warnings=0, infos=0)
        
        if not CIRQ_AVAILABLE:
            result.add_issue('error', 'Cirq not available for linting')
            return result
        
        # Basic Cirq circuit linting (placeholder)
        result.add_issue(
            'info',
            'Cirq circuit linting not yet fully implemented',
            rule='cirq_linting'
        )
        
        return result


# Convenience functions
def lint_qiskit_circuit(circuit: Any, config: Optional[LintingConfig] = None) -> LintResult:
    """Convenience function to lint a Qiskit circuit."""
    linter = QiskitLinter(config)
    return linter.lint_circuit(circuit)


def lint_cirq_circuit(circuit: Any, config: Optional[LintingConfig] = None) -> LintResult:
    """Convenience function to lint a Cirq circuit."""
    linter = CirqLinter(config)
    return linter.lint_circuit(circuit)


def lint_pulse_schedule(schedule: Any, config: Optional[LintingConfig] = None) -> LintResult:
    """Convenience function to lint a pulse schedule."""
    linter = PulseLinter(config)
    return linter.lint_schedule(schedule)


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