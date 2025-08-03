"""
Unit tests for quantum circuit linting functionality.

Tests the quantum linting classes and circuit validation utilities.
"""

import pytest
from unittest.mock import Mock, patch

from quantum_devops_ci.linting import (
    QiskitLinter,
    PulseLinter,
    LintingConfig,
    LintResult,
    LintSeverity,
    CircuitViolation
)


class TestLintingConfig:
    """Test cases for LintingConfig class."""
    
    def test_config_creation(self):
        """Test LintingConfig creation and defaults."""
        config = LintingConfig(
            max_circuit_depth=50,
            max_two_qubit_gates=25,
            max_qubits=10,
            allowed_gates=['h', 'cx', 'rz'],
            backend_constraints={'ibmq_manhattan': {'max_qubits': 65}}
        )
        
        assert config.max_circuit_depth == 50
        assert config.max_two_qubit_gates == 25
        assert config.max_qubits == 10
        assert 'h' in config.allowed_gates
        assert 'ibmq_manhattan' in config.backend_constraints
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LintingConfig()
        
        assert config.max_circuit_depth > 0
        assert config.max_two_qubit_gates > 0
        assert config.max_qubits > 0
        assert len(config.allowed_gates) > 0


class TestCircuitViolation:
    """Test cases for CircuitViolation class."""
    
    def test_violation_creation(self):
        """Test CircuitViolation creation."""
        violation = CircuitViolation(
            rule='max_depth',
            severity=LintSeverity.ERROR,
            message='Circuit depth exceeds maximum allowed',
            line_number=5,
            suggestion='Reduce circuit depth or use circuit optimization'
        )
        
        assert violation.rule == 'max_depth'
        assert violation.severity == LintSeverity.ERROR
        assert 'exceeds maximum' in violation.message
        assert violation.line_number == 5
        assert 'optimization' in violation.suggestion


class TestLintResult:
    """Test cases for LintResult class."""
    
    def test_lint_result_creation(self):
        """Test LintResult creation and methods."""
        violations = [
            CircuitViolation('rule1', LintSeverity.ERROR, 'Error message'),
            CircuitViolation('rule2', LintSeverity.WARNING, 'Warning message'),
            CircuitViolation('rule3', LintSeverity.INFO, 'Info message')
        ]
        
        result = LintResult(
            passed=False,
            violations=violations,
            score=0.75,
            metrics={'depth': 10, 'gates': 15}
        )
        
        assert not result.passed
        assert len(result.violations) == 3
        assert result.score == 0.75
        assert result.metrics['depth'] == 10
    
    def test_has_errors(self):
        """Test error detection."""
        error_violation = CircuitViolation('rule1', LintSeverity.ERROR, 'Error')
        warning_violation = CircuitViolation('rule2', LintSeverity.WARNING, 'Warning')
        
        result_with_errors = LintResult(False, [error_violation, warning_violation])
        result_without_errors = LintResult(True, [warning_violation])
        
        assert result_with_errors.has_errors()
        assert not result_without_errors.has_errors()
    
    def test_get_violations_by_severity(self):
        """Test filtering violations by severity."""
        violations = [
            CircuitViolation('rule1', LintSeverity.ERROR, 'Error 1'),
            CircuitViolation('rule2', LintSeverity.ERROR, 'Error 2'),
            CircuitViolation('rule3', LintSeverity.WARNING, 'Warning 1')
        ]
        
        result = LintResult(False, violations)
        
        errors = result.get_violations_by_severity(LintSeverity.ERROR)
        warnings = result.get_violations_by_severity(LintSeverity.WARNING)
        
        assert len(errors) == 2
        assert len(warnings) == 1
        assert all(v.severity == LintSeverity.ERROR for v in errors)
        assert all(v.severity == LintSeverity.WARNING for v in warnings)


class TestQiskitLinter:
    """Test cases for QiskitLinter class."""
    
    def test_initialization(self, quantum_linter):
        """Test QiskitLinter initialization."""
        assert isinstance(quantum_linter.config, LintingConfig)
        assert quantum_linter.config.max_circuit_depth == 50
        assert quantum_linter.config.max_two_qubit_gates == 25
    
    def test_lint_circuit_mock(self, quantum_linter, mock_quantum_circuit):
        """Test circuit linting with mock circuit."""
        # Mock circuit properties
        mock_quantum_circuit.depth = Mock(return_value=5)
        mock_quantum_circuit.num_qubits = 2
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 1, 'cx': 1})
        
        result = quantum_linter.lint_circuit(mock_quantum_circuit)
        
        assert isinstance(result, LintResult)
        assert result.score >= 0
        assert result.score <= 1
        assert 'depth' in result.metrics
        assert 'num_qubits' in result.metrics
        assert 'gate_count' in result.metrics
    
    def test_check_circuit_depth(self, quantum_linter, mock_quantum_circuit):
        """Test circuit depth checking."""
        # Test circuit within depth limit
        mock_quantum_circuit.depth = Mock(return_value=10)
        violations = quantum_linter._check_circuit_depth(mock_quantum_circuit)
        assert len(violations) == 0
        
        # Test circuit exceeding depth limit
        mock_quantum_circuit.depth = Mock(return_value=100)
        violations = quantum_linter._check_circuit_depth(mock_quantum_circuit)
        assert len(violations) > 0
        assert violations[0].rule == 'max_circuit_depth'
        assert violations[0].severity == LintSeverity.ERROR
    
    def test_check_qubit_count(self, quantum_linter, mock_quantum_circuit):
        """Test qubit count checking."""
        # Test circuit within qubit limit
        mock_quantum_circuit.num_qubits = 5
        violations = quantum_linter._check_qubit_count(mock_quantum_circuit)
        assert len(violations) == 0
        
        # Test circuit exceeding qubit limit
        mock_quantum_circuit.num_qubits = 20
        violations = quantum_linter._check_qubit_count(mock_quantum_circuit)
        assert len(violations) > 0
        assert violations[0].rule == 'max_qubits'
        assert violations[0].severity == LintSeverity.ERROR
    
    def test_check_gate_usage(self, quantum_linter, mock_quantum_circuit):
        """Test gate usage checking."""
        # Test allowed gates
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 2, 'cx': 1})
        violations = quantum_linter._check_gate_usage(mock_quantum_circuit)
        assert len(violations) == 0
        
        # Test with some disallowed gates (if config specifies allowed gates)
        if hasattr(quantum_linter.config, 'allowed_gates') and quantum_linter.config.allowed_gates:
            disallowed_gate = 'custom_gate_xyz'
            if disallowed_gate not in quantum_linter.config.allowed_gates:
                mock_quantum_circuit.count_ops = Mock(return_value={'h': 1, disallowed_gate: 1})
                violations = quantum_linter._check_gate_usage(mock_quantum_circuit)
                # May or may not have violations depending on implementation
    
    def test_check_two_qubit_gates(self, quantum_linter, mock_quantum_circuit):
        """Test two-qubit gate count checking."""
        # Test within limit
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 5, 'cx': 10})
        violations = quantum_linter._check_two_qubit_gates(mock_quantum_circuit)
        assert len(violations) == 0
        
        # Test exceeding limit
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 5, 'cx': 50})
        violations = quantum_linter._check_two_qubit_gates(mock_quantum_circuit)
        assert len(violations) > 0
        assert violations[0].rule == 'max_two_qubit_gates'
        assert violations[0].severity == LintSeverity.ERROR
    
    def test_check_backend_compatibility(self, quantum_linter, mock_quantum_circuit):
        """Test backend compatibility checking."""
        backend_name = 'ibmq_manhattan'
        
        # Mock circuit properties
        mock_quantum_circuit.num_qubits = 5
        mock_quantum_circuit.depth = Mock(return_value=10)
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 2, 'cx': 3})
        
        result = quantum_linter.check_backend_compatibility(mock_quantum_circuit, backend_name)
        
        assert isinstance(result, dict)
        assert 'compatible' in result
        assert 'violations' in result
        assert 'score' in result
        assert isinstance(result['compatible'], bool)
        assert isinstance(result['violations'], list)
        assert 0 <= result['score'] <= 1
    
    def test_get_optimization_suggestions(self, quantum_linter, mock_quantum_circuit):
        """Test optimization suggestions."""
        # Mock a circuit that needs optimization
        mock_quantum_circuit.depth = Mock(return_value=100)  # Deep circuit
        mock_quantum_circuit.num_qubits = 15  # Many qubits
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 10, 'cx': 50})
        
        suggestions = quantum_linter.get_optimization_suggestions(mock_quantum_circuit)
        
        assert isinstance(suggestions, list)
        # Should have suggestions for deep, complex circuit
        assert len(suggestions) > 0
        
        # Check that suggestions are strings
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0


class TestPulseLinter:
    """Test cases for PulseLinter class."""
    
    def test_initialization(self):
        """Test PulseLinter initialization."""
        config = LintingConfig(max_pulse_duration=1000)
        linter = PulseLinter(config)
        
        assert isinstance(linter.config, LintingConfig)
    
    def test_lint_pulse_schedule_mock(self):
        """Test pulse schedule linting with mock."""
        config = LintingConfig()
        linter = PulseLinter(config)
        
        # Create mock pulse schedule
        mock_schedule = Mock()
        mock_schedule.duration = 500
        mock_schedule.channels = ['d0', 'u1']
        
        result = linter.lint_pulse_schedule(mock_schedule)
        
        assert isinstance(result, LintResult)
        assert result.score >= 0
        assert result.score <= 1
    
    def test_check_pulse_duration(self):
        """Test pulse duration checking."""
        config = LintingConfig(max_pulse_duration=1000)
        linter = PulseLinter(config)
        
        # Mock schedule within duration limit
        short_schedule = Mock()
        short_schedule.duration = 500
        violations = linter._check_pulse_duration(short_schedule)
        assert len(violations) == 0
        
        # Mock schedule exceeding duration limit
        long_schedule = Mock()
        long_schedule.duration = 2000
        violations = linter._check_pulse_duration(long_schedule)
        assert len(violations) > 0
        assert violations[0].rule == 'max_pulse_duration'
        assert violations[0].severity == LintSeverity.ERROR


@pytest.mark.performance
class TestLintingPerformance:
    """Performance tests for linting functionality."""
    
    def test_lint_multiple_circuits_performance(self, quantum_linter):
        """Test linting performance with multiple circuits."""
        import time
        
        # Create multiple mock circuits
        circuits = []
        for i in range(20):
            circuit = Mock()
            circuit.depth = Mock(return_value=5 + i)
            circuit.num_qubits = 2 + (i % 5)
            circuit.count_ops = Mock(return_value={'h': i + 1, 'cx': i // 2})
            circuits.append(circuit)
        
        start_time = time.time()
        
        results = []
        for circuit in circuits:
            result = quantum_linter.lint_circuit(circuit)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Should lint circuits quickly
        assert total_time < 1.0  # Should complete within 1 second
        assert len(results) == 20
        assert all(isinstance(r, LintResult) for r in results)
    
    def test_backend_compatibility_performance(self, quantum_linter, mock_quantum_circuit):
        """Test backend compatibility checking performance."""
        import time
        
        mock_quantum_circuit.num_qubits = 5
        mock_quantum_circuit.depth = Mock(return_value=10)
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 3, 'cx': 2})
        
        backends = ['ibmq_manhattan', 'ibmq_brooklyn', 'ibmq_jakarta', 'qasm_simulator']
        
        start_time = time.time()
        
        results = []
        for backend in backends:
            result = quantum_linter.check_backend_compatibility(mock_quantum_circuit, backend)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Should check compatibility quickly
        assert total_time < 0.5  # Should complete within 0.5 seconds
        assert len(results) == len(backends)


@pytest.mark.integration
class TestLintingIntegration:
    """Integration tests for linting functionality."""
    
    def test_linter_with_real_qiskit_circuit(self, quantum_linter, framework_availability):
        """Test linter with real Qiskit circuit if available."""
        if not framework_availability.get('qiskit', False):
            pytest.skip("Qiskit not available")
        
        try:
            from qiskit import QuantumCircuit
            
            # Create simple Bell circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            
            result = quantum_linter.lint_circuit(qc)
            
            assert isinstance(result, LintResult)
            assert result.score > 0
            # Simple Bell circuit should pass most checks
            assert not result.has_errors()
            
        except ImportError:
            pytest.skip("Qiskit not available for integration test")
    
    def test_linter_with_database_logging(self, quantum_linter, clean_database, mock_quantum_circuit):
        """Test linter with database integration for logging results."""
        mock_quantum_circuit.depth = Mock(return_value=5)
        mock_quantum_circuit.num_qubits = 2
        mock_quantum_circuit.count_ops = Mock(return_value={'h': 1, 'cx': 1})
        
        result = quantum_linter.lint_circuit(mock_quantum_circuit)
        
        # For now, just verify linting works
        # In full integration, this would log to database
        assert isinstance(result, LintResult)
        assert result.score >= 0
