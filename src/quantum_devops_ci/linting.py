"""
Quantum circuit linting and validation tools.

This module provides linting capabilities for quantum circuits, including
pulse-level analysis, gate constraint checking, and hardware compatibility
validation.
"""

import abc
import warnings
import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary for serialization."""
        return {
            'severity': self.severity,
            'message': self.message,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column': self.column,
            'rule': self.rule,
            'suggestion': self.suggestion,
            'time': self.time
        }

    def __str__(self) -> str:
        """String representation of the issue."""
        parts = []
        if self.file_path:
            parts.append(f"{self.file_path}")
            if self.line_number:
                parts[-1] += f":{self.line_number}"
                if self.column:
                    parts[-1] += f":{self.column}"
        
        severity_symbol = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(self.severity, '?')
        parts.append(f"{severity_symbol} {self.severity.upper()}")
        
        if self.rule:
            parts.append(f"[{self.rule}]")
        
        parts.append(self.message)
        
        if self.suggestion:
            parts.append(f"\n  💡 Suggestion: {self.suggestion}")
        
        return " ".join(parts)


@dataclass
class LintResult:
    """Container for linting results."""
    issues: List[LintIssue] = field(default_factory=list)
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    
    def __post_init__(self):
        self._recalculate_stats()
    
    def _recalculate_stats(self):
        """Recalculate statistics from issues."""
        self.total_issues = len(self.issues)
        self.errors = sum(1 for issue in self.issues if issue.severity == 'error')
        self.warnings = sum(1 for issue in self.issues if issue.severity == 'warning')
        self.infos = sum(1 for issue in self.issues if issue.severity == 'info')
    
    def has_errors(self) -> bool:
        """Check if linting found any errors."""
        return self.errors > 0
    
    def has_warnings(self) -> bool:
        """Check if linting found any warnings."""
        return self.warnings > 0
    
    def add_issue(self, severity: str, message: str, **kwargs):
        """Add a linting issue."""
        issue = LintIssue(severity=severity, message=message, **kwargs)
        self.issues.append(issue)
        self._recalculate_stats()
    
    def merge(self, other: 'LintResult'):
        """Merge another lint result into this one."""
        self.issues.extend(other.issues)
        self._recalculate_stats()
    
    def filter_by_severity(self, severities: List[str]) -> 'LintResult':
        """Filter issues by severity levels."""
        filtered_issues = [issue for issue in self.issues if issue.severity in severities]
        result = LintResult(issues=filtered_issues)
        return result
    
    def get_summary(self) -> str:
        """Get a summary string of linting results."""
        if self.total_issues == 0:
            return "✅ No issues found"
        
        parts = []
        if self.errors > 0:
            parts.append(f"❌ {self.errors} error{'s' if self.errors != 1 else ''}")
        if self.warnings > 0:
            parts.append(f"⚠️ {self.warnings} warning{'s' if self.warnings != 1 else ''}")
        if self.infos > 0:
            parts.append(f"ℹ️ {self.infos} info")
        
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'total_issues': self.total_issues,
            'errors': self.errors,
            'warnings': self.warnings,
            'infos': self.infos,
            'summary': self.get_summary(),
            'issues': [issue.to_dict() for issue in self.issues]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class LintingConfig:
    """Configuration for quantum circuit linting."""
    max_circuit_depth: int = 100
    max_two_qubit_gates: int = 50
    max_qubits: int = 100
    allowed_gates: List[str] = field(default_factory=lambda: ['u1', 'u2', 'u3', 'cx'])
    forbidden_gates: List[str] = field(default_factory=list)
    pulse_constraints: Optional[Dict[str, Any]] = None
    gate_constraints: Optional[Dict[str, Any]] = None
    optimization_level: int = 1
    check_connectivity: bool = True
    connectivity_graph: Optional[Dict[str, List[int]]] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'LintingConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            warnings.warn("YAML not available - using default configuration")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract relevant sections
            pulse_config = config_data.get('pulse_constraints', {})
            gate_config = config_data.get('gate_constraints', {})
            
            # Handle allowed_gates format
            allowed_gates = gate_config.get('allowed_gates', ['u1', 'u2', 'u3', 'cx'])
            if isinstance(allowed_gates, list) and all(isinstance(item, dict) for item in allowed_gates):
                # Extract gate names from complex format
                allowed_gates = [item['name'] for item in allowed_gates if 'name' in item]
            
            return cls(
                max_circuit_depth=gate_config.get('max_circuit_depth', 100),
                max_two_qubit_gates=gate_config.get('max_two_qubit_gates', 50),
                max_qubits=gate_config.get('max_qubits', 100),
                allowed_gates=allowed_gates,
                forbidden_gates=gate_config.get('forbidden_gates', []),
                pulse_constraints=pulse_config,
                gate_constraints=gate_config,
                check_connectivity=gate_config.get('check_connectivity', True)
            )
        except FileNotFoundError:
            warnings.warn(f"Configuration file not found: {config_path}, using defaults")
            return cls()
        except Exception as e:
            warnings.warn(f"Failed to load config from {config_path}: {e}, using defaults")
            return cls()


class QuantumLinter(abc.ABC):
    """Abstract base class for quantum circuit linters."""
    
    def __init__(self, config: Optional[Union[str, LintingConfig]] = None):
        """Initialize linter with configuration."""
        if isinstance(config, str):
            self.config = LintingConfig.from_file(config)
        elif isinstance(config, LintingConfig):
            self.config = config
        else:
            self.config = LintingConfig()
        
        self.logger = logging.getLogger(__name__)
    
    @abc.abstractmethod
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint a quantum circuit."""
        pass
    
    def lint_file(self, file_path: str) -> LintResult:
        """Lint circuits in a file."""
        result = LintResult()
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                result.add_issue('error', f'File not found: {file_path}', file_path=file_path)
                return result
            
            # Check file extension
            if file_path_obj.suffix not in ['.py', '.qasm', '.qpy']:
                result.add_issue('warning', f'Unknown file type: {file_path_obj.suffix}', file_path=file_path)
            
            # For Python files, look for quantum circuit definitions
            if file_path_obj.suffix == '.py':
                result = self._lint_python_file(file_path)
            elif file_path_obj.suffix == '.qasm':
                result = self._lint_qasm_file(file_path)
            else:
                result.add_issue('info', f'File linting not fully implemented for {file_path}', 
                               file_path=file_path)
        
        except PermissionError:
            result.add_issue('error', f'Permission denied accessing file: {file_path}', file_path=file_path)
        except Exception as e:
            result.add_issue('error', f'Failed to lint file {file_path}: {e}', file_path=file_path)
        
        return result
    
    def _lint_python_file(self, file_path: str) -> LintResult:
        """Lint Python file for quantum circuits."""
        result = LintResult()
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic checks for quantum code patterns
            if 'QuantumCircuit' in content:
                result.add_issue('info', 'Found Qiskit QuantumCircuit usage', file_path=file_path)
            
            if 'cirq.Circuit' in content:
                result.add_issue('info', 'Found Cirq Circuit usage', file_path=file_path)
            
            # Check for common issues
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for hardcoded quantum parameters
                if 'shots=' in line_stripped and any(char.isdigit() for char in line_stripped):
                    result.add_issue('warning', 'Hardcoded shots parameter found', 
                                   file_path=file_path, line_number=i,
                                   rule='hardcoded_parameters',
                                   suggestion='Consider using configuration or constants')
                
                # Check for missing error handling
                if '.run(' in line_stripped and 'try:' not in content[:content.find(line)]:
                    result.add_issue('warning', 'Quantum execution without error handling', 
                                   file_path=file_path, line_number=i,
                                   rule='error_handling',
                                   suggestion='Wrap quantum execution in try-except block')
        
        except Exception as e:
            result.add_issue('error', f'Error analyzing Python file: {e}', file_path=file_path)
        
        return result
    
    def _lint_qasm_file(self, file_path: str) -> LintResult:
        """Lint QASM file."""
        result = LintResult()
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Basic QASM validation
            if not any(line.strip().startswith('OPENQASM') for line in lines):
                result.add_issue('error', 'Missing OPENQASM version declaration', file_path=file_path)
            
            # Check for gate definitions and usage
            gate_count = sum(1 for line in lines if line.strip() and not line.strip().startswith('//'))
            
            if gate_count > self.config.max_circuit_depth:
                result.add_issue('warning', f'QASM file has {gate_count} operations, exceeding depth limit',
                               file_path=file_path, rule='max_circuit_depth')
            
        except Exception as e:
            result.add_issue('error', f'Error analyzing QASM file: {e}', file_path=file_path)
        
        return result
    
    def lint_directory(self, directory_path: str) -> LintResult:
        """Lint all quantum files in a directory."""
        result = LintResult()
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                result.add_issue('error', f'Directory not found: {directory_path}')
                return result
            
            if not directory.is_dir():
                result.add_issue('error', f'Path is not a directory: {directory_path}')
                return result
            
            # Find quantum files
            quantum_patterns = ['**/*.py', '**/*.qasm', '**/*.qpy']
            quantum_files = []
            
            for pattern in quantum_patterns:
                quantum_files.extend(directory.glob(pattern))
            
            if not quantum_files:
                result.add_issue('warning', f'No quantum files found in {directory_path}')
                return result
            
            self.logger.info(f"Linting {len(quantum_files)} files in {directory_path}")
            
            for file_path in quantum_files:
                try:
                    file_result = self.lint_file(str(file_path))
                    result.merge(file_result)
                except Exception as e:
                    result.add_issue('error', f'Failed to lint {file_path}: {e}')
            
        except PermissionError:
            result.add_issue('error', f'Permission denied accessing directory: {directory_path}')
        except Exception as e:
            result.add_issue('error', f'Failed to lint directory {directory_path}: {e}')
        
        return result


class QiskitLinter(QuantumLinter):
    """Linter for Qiskit quantum circuits."""
    
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint a Qiskit quantum circuit."""
        result = LintResult()
        
        if not QISKIT_AVAILABLE:
            result.add_issue('error', 'Qiskit not available for linting')
            return result
        
        try:
            # Validate circuit object
            if not hasattr(circuit, 'data') or not hasattr(circuit, 'num_qubits'):
                result.add_issue('error', 'Invalid Qiskit circuit object - missing required attributes')
                return result
            
            self.logger.info(f"Linting Qiskit circuit with {circuit.num_qubits} qubits")
            
            # Basic circuit validation
            self._check_basic_properties(circuit, result)
            
            # Gate analysis
            self._check_gate_usage(circuit, result)
            
            # Connectivity analysis
            if self.config.check_connectivity:
                self._check_connectivity(circuit, result)
            
            # Performance checks
            self._check_performance_characteristics(circuit, result)
            
            # Measurement checks
            self._check_measurements(circuit, result)
            
        except Exception as e:
            result.add_issue('error', f'Unexpected error during circuit linting: {e}')
        
        return result
    
    def _check_basic_properties(self, circuit: Any, result: LintResult):
        """Check basic circuit properties."""
        try:
            # Check circuit depth
            depth = circuit.depth()
            if depth > self.config.max_circuit_depth:
                result.add_issue(
                    'warning',
                    f'Circuit depth ({depth}) exceeds recommended maximum ({self.config.max_circuit_depth})',
                    rule='max_circuit_depth',
                    suggestion='Consider circuit optimization or breaking into smaller subcircuits'
                )
            
            # Check number of qubits
            if circuit.num_qubits > self.config.max_qubits:
                result.add_issue(
                    'error',
                    f'Circuit uses {circuit.num_qubits} qubits, exceeding limit of {self.config.max_qubits}',
                    rule='max_qubits',
                    suggestion='Reduce circuit size or adjust configuration limits'
                )
            
            # Check for empty circuit
            if len(circuit.data) == 0:
                result.add_issue(
                    'warning',
                    'Circuit is empty (no gates or operations)',
                    rule='empty_circuit'
                )
            
        except Exception as e:
            result.add_issue('error', f'Error checking basic properties: {e}')
    
    def _check_gate_usage(self, circuit: Any, result: LintResult):
        """Check gate usage patterns."""
        try:
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
                count = forbidden_gates.count(gate)
                result.add_issue(
                    'error',
                    f'Forbidden gate used: {gate} ({count} times)',
                    rule='forbidden_gates',
                    suggestion=f'Replace {gate} with allowed gate decomposition'
                )
            
            # Report unknown gates
            for gate in set(unknown_gates):
                count = unknown_gates.count(gate)
                result.add_issue(
                    'warning',
                    f'Unknown/non-standard gate: {gate} ({count} times)',
                    rule='allowed_gates',
                    suggestion=f'Verify {gate} is supported by target backend'
                )
            
            # Check two-qubit gate count
            two_qubit_gates = ['cx', 'cy', 'cz', 'cnot', 'cphase', 'crz', 'cu', 'cu1', 'cu3']
            two_qubit_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
            
            if two_qubit_count > self.config.max_two_qubit_gates:
                result.add_issue(
                    'warning',
                    f'High two-qubit gate count ({two_qubit_count}) may impact fidelity',
                    rule='max_two_qubit_gates',
                    suggestion='Consider circuit optimization to reduce two-qubit gates'
                )
            
            # Check for gate optimization opportunities
            if gate_counts.get('u1', 0) > 10:
                result.add_issue(
                    'info',
                    f'Many U1 gates ({gate_counts["u1"]}) - consider virtual Z gate optimization',
                    rule='gate_optimization'
                )
            
        except Exception as e:
            result.add_issue('error', f'Error analyzing gate usage: {e}')
    
    def _check_connectivity(self, circuit: Any, result: LintResult):
        """Check circuit connectivity constraints."""
        try:
            connectivity_violations = 0
            
            for instruction in circuit.data:
                if len(instruction.qubits) == 2:
                    qubit1 = instruction.qubits[0].index
                    qubit2 = instruction.qubits[1].index
                    
                    # Check if qubits are connected (simplified linear connectivity by default)
                    if self.config.connectivity_graph:
                        # Use custom connectivity graph if provided
                        if qubit2 not in self.config.connectivity_graph.get(qubit1, []):
                            connectivity_violations += 1
                            result.add_issue(
                                'error',
                                f'Non-connected qubits: {instruction.operation.name} on qubits {qubit1}, {qubit2}',
                                rule='connectivity',
                                suggestion='Use connected qubits or add SWAP gates'
                            )
                    else:
                        # Default: assume linear connectivity (adjacent qubits only)
                        if abs(qubit1 - qubit2) > 1:
                            connectivity_violations += 1
                            result.add_issue(
                                'warning',
                                f'Non-adjacent qubit gate: {instruction.operation.name} on qubits {qubit1}, {qubit2}',
                                rule='connectivity',
                                suggestion='Consider SWAP gates or circuit routing optimization'
                            )
            
            if connectivity_violations == 0:
                result.add_issue(
                    'info',
                    'All two-qubit gates respect connectivity constraints',
                    rule='connectivity'
                )
            
        except Exception as e:
            result.add_issue('error', f'Error checking connectivity: {e}')
    
    def _check_performance_characteristics(self, circuit: Any, result: LintResult):
        """Check performance-related characteristics."""
        try:
            # Check for barriers
            barrier_count = sum(1 for inst in circuit.data if inst.operation.name == 'barrier')
            if barrier_count > circuit.num_qubits:
                result.add_issue(
                    'warning',
                    f'Excessive barriers ({barrier_count}) may impact optimization',
                    rule='performance',
                    suggestion='Remove unnecessary barriers to improve transpilation'
                )
            
            # Check circuit width vs depth ratio
            if circuit.num_qubits > 0:
                width_depth_ratio = circuit.depth() / circuit.num_qubits
                if width_depth_ratio > 50:
                    result.add_issue(
                        'warning',
                        f'High depth-to-width ratio ({width_depth_ratio:.1f}) may indicate inefficient design',
                        rule='performance',
                        suggestion='Consider parallelizing operations or reducing circuit depth'
                    )
            
        except Exception as e:
            result.add_issue('error', f'Error checking performance characteristics: {e}')
    
    def _check_measurements(self, circuit: Any, result: LintResult):
        """Check measurement operations."""
        try:
            measurement_count = sum(1 for inst in circuit.data if inst.operation.name == 'measure')
            
            if measurement_count == 0:
                result.add_issue(
                    'info',
                    'Circuit has no measurement operations',
                    rule='measurements',
                    suggestion='Add measurements if this circuit will be executed on hardware'
                )
            elif measurement_count > circuit.num_qubits:
                result.add_issue(
                    'warning',
                    f'More measurements ({measurement_count}) than qubits ({circuit.num_qubits})',
                    rule='measurements',
                    suggestion='Verify measurement configuration'
                )
            
        except Exception as e:
            result.add_issue('error', f'Error checking measurements: {e}')


class PulseLinter(QuantumLinter):
    """Linter for quantum pulse schedules."""
    
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint circuit for pulse-level constraints."""
        result = LintResult()
        
        # Pulse linting would require pulse schedule objects
        result.add_issue(
            'info',
            'Pulse-level circuit linting not yet fully implemented',
            rule='pulse_linting',
            suggestion='Use lint_schedule() method for pulse schedule objects'
        )
        
        return result
    
    def lint_schedule(self, schedule: Any) -> LintResult:
        """Lint a pulse schedule."""
        result = LintResult()
        
        try:
            if not self.config.pulse_constraints:
                result.add_issue('warning', 'No pulse constraints configured')
                return result
            
            pulse_config = self.config.pulse_constraints
            
            # Check pulse amplitude constraints
            max_amplitude = pulse_config.get('max_amplitude', 1.0)
            min_duration = pulse_config.get('min_pulse_duration', 16)
            
            self.logger.info(f"Checking pulse schedule against constraints")
            
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
            
            # Phase coherence check
            phase_granularity = pulse_config.get('phase_granularity', 0.01)
            result.add_issue(
                'info',
                f'Phase granularity constraint: {phase_granularity} radians',
                rule='phase_granularity'
            )
            
        except Exception as e:
            result.add_issue('error', f'Error linting pulse schedule: {e}')
        
        return result


class CirqLinter(QuantumLinter):
    """Linter for Cirq quantum circuits."""
    
    def lint_circuit(self, circuit: Any) -> LintResult:
        """Lint a Cirq quantum circuit."""
        result = LintResult()
        
        if not CIRQ_AVAILABLE:
            result.add_issue('error', 'Cirq not available for linting')
            return result
        
        try:
            # Basic Cirq circuit linting
            if hasattr(circuit, 'moments'):
                moment_count = len(circuit.moments)
                if moment_count > self.config.max_circuit_depth:
                    result.add_issue(
                        'warning',
                        f'Circuit has {moment_count} moments, exceeding depth limit',
                        rule='max_circuit_depth'
                    )
                
                # Check qubit usage
                all_qubits = set()
                for moment in circuit.moments:
                    for operation in moment.operations:
                        all_qubits.update(operation.qubits)
                
                if len(all_qubits) > self.config.max_qubits:
                    result.add_issue(
                        'error',
                        f'Circuit uses {len(all_qubits)} qubits, exceeding limit',
                        rule='max_qubits'
                    )
                
                result.add_issue(
                    'info',
                    f'Cirq circuit analyzed: {moment_count} moments, {len(all_qubits)} qubits',
                    rule='cirq_analysis'
                )
            else:
                result.add_issue(
                    'error',
                    'Invalid Cirq circuit - missing moments attribute',
                    rule='cirq_validation'
                )
        
        except Exception as e:
            result.add_issue('error', f'Error linting Cirq circuit: {e}')
        
        return result


class LintingCLI:
    """Command-line interface for quantum linting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run(self, framework: str, target: str, config_path: Optional[str] = None, output_format: str = 'text') -> int:
        """Run linting command."""
        try:
            # Load configuration
            config = None
            if config_path:
                config = LintingConfig.from_file(config_path)
            else:
                config = LintingConfig()
            
            # Create appropriate linter
            if framework == 'qiskit':
                linter = QiskitLinter(config)
            elif framework == 'cirq':
                linter = CirqLinter(config)
            elif framework == 'pulse':
                linter = PulseLinter(config)
            else:
                print(f"❌ Unknown framework: {framework}")
                return 1
            
            # Determine if target is file or directory
            target_path = Path(target)
            
            if target_path.is_file():
                result = linter.lint_file(target)
            elif target_path.is_dir():
                result = linter.lint_directory(target)
            else:
                print(f"❌ Target not found: {target}")
                return 1
            
            # Output results
            if output_format == 'json':
                print(result.to_json())
            else:
                print(f"\n🔍 Quantum Linting Results for {target}")
                print("=" * 50)
                print(result.get_summary())
                print()
                
                if result.issues:
                    for issue in result.issues:
                        print(issue)
                        print()
            
            # Return appropriate exit code
            return 1 if result.has_errors() else 0
        
        except Exception as e:
            print(f"❌ Linting failed: {e}")
            return 1


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
    import sys
    
    parser = argparse.ArgumentParser(description='Quantum circuit linter')
    parser.add_argument('target', help='File or directory to lint')
    parser.add_argument('--framework', choices=['qiskit', 'cirq', 'pulse'], default='qiskit',
                       help='Quantum framework to use')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    
    args = parser.parse_args()
    
    cli = LintingCLI()
    exit_code = cli.run(args.framework, args.target, args.config, args.format)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()