"""
Test suite for validation and security modules.
"""

import pytest
import tempfile
import os
from pathlib import Path

from quantum_devops_ci.validation import (
    SecurityValidator, ConfigValidator, QuantumCircuitValidator,
    InputSanitizer
)
from quantum_devops_ci.exceptions import (
    SecurityError, ConfigurationError, CircuitValidationError
)


class TestSecurityValidator:
    """Test security validation functions."""
    
    def test_validate_code_string_safe(self):
        """Test that safe code passes validation."""
        safe_code = "import qiskit\nqc = QuantumCircuit(2)\nqc.h(0)"
        assert SecurityValidator.validate_code_string(safe_code) is True
    
    def test_validate_code_string_dangerous(self):
        """Test that dangerous code is rejected."""
        dangerous_patterns = [
            "__import__('os').system('ls')",
            "eval('print(1)')",
            "exec('import os')",
            "open('/etc/passwd')",
            "subprocess.call(['ls'])",
            "os.system('whoami')"
        ]
        
        for pattern in dangerous_patterns:
            with pytest.raises(SecurityError):
                SecurityValidator.validate_code_string(pattern)
    
    def test_validate_file_path_safe(self):
        """Test that safe file paths pass validation."""
        safe_paths = [
            "/home/user/circuits/test.py",
            "circuits/quantum_test.qasm",
            "./test_data/circuit.json"
        ]
        
        for path in safe_paths:
            assert SecurityValidator.validate_file_path(path) is True
    
    def test_validate_file_path_dangerous(self):
        """Test that dangerous file paths are rejected."""
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "/bin/bash",
            "../../config/secret.key"
        ]
        
        for path in dangerous_paths:
            with pytest.raises(SecurityError):
                SecurityValidator.validate_file_path(path)
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        # Test normal string
        result = SecurityValidator.sanitize_string("hello world")
        assert result == "hello world"
        
        # Test string with control characters
        result = SecurityValidator.sanitize_string("hello\x00\x01world")
        assert result == "helloworld"
        
        # Test length limiting
        long_string = "a" * 2000
        result = SecurityValidator.sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    def test_validate_json_input(self):
        """Test JSON validation."""
        # Valid JSON
        valid_json = '{"shots": 1000, "backend": "qasm_simulator"}'
        result = SecurityValidator.validate_json_input(valid_json)
        assert result == {"shots": 1000, "backend": "qasm_simulator"}
        
        # Invalid JSON
        with pytest.raises(SecurityError):
            SecurityValidator.validate_json_input('{"invalid": json}')
        
        # Too large JSON
        large_json = '{"data": "' + "x" * 2_000_000 + '"}'
        with pytest.raises(SecurityError):
            SecurityValidator.validate_json_input(large_json)


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_validate_testing_config_valid(self):
        """Test valid testing configuration."""
        config = {
            'noise_model': {'depolarizing': 0.01},
            'backend_preferences': ['qasm_simulator', 'statevector_simulator'],
            'tolerance': 0.1
        }
        
        assert ConfigValidator.validate_config(config, 'testing_config') is True
    
    def test_validate_testing_config_invalid(self):
        """Test invalid testing configuration."""
        # Missing required field
        config = {
            'noise_model': {'depolarizing': 0.01},
            'backend_preferences': ['qasm_simulator']
            # Missing 'tolerance'
        }
        
        with pytest.raises(ConfigurationError):
            ConfigValidator.validate_config(config, 'testing_config')
        
        # Invalid tolerance value
        config = {
            'noise_model': {'depolarizing': 0.01},
            'backend_preferences': ['qasm_simulator'],
            'tolerance': 1.5  # > 1.0
        }
        
        with pytest.raises(ConfigurationError):
            ConfigValidator.validate_config(config, 'testing_config')
    
    def test_validate_scheduling_config(self):
        """Test scheduling configuration validation."""
        # Valid config
        config = {
            'optimization_goal': 'minimize_cost',
            'max_queue_time': 300
        }
        assert ConfigValidator.validate_config(config, 'scheduling_config') is True
        
        # Invalid optimization goal
        config = {
            'optimization_goal': 'invalid_goal',
            'max_queue_time': 300
        }
        with pytest.raises(ConfigurationError):
            ConfigValidator.validate_config(config, 'scheduling_config')


class TestQuantumCircuitValidator:
    """Test quantum circuit validation."""
    
    def test_validate_circuit_parameters_valid(self):
        """Test valid circuit parameters."""
        assert QuantumCircuitValidator.validate_circuit_parameters(5, 10, 8) is True
        assert QuantumCircuitValidator.validate_circuit_parameters(1, 1, 1) is True
    
    def test_validate_circuit_parameters_invalid(self):
        """Test invalid circuit parameters."""
        # Too many qubits
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_circuit_parameters(200, 10, 8)
        
        # Too many gates
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_circuit_parameters(5, 20000, 8)
        
        # Too deep
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_circuit_parameters(5, 10, 2000)
        
        # Negative values
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_circuit_parameters(-1, 10, 8)
    
    def test_validate_shots(self):
        """Test shots validation."""
        assert QuantumCircuitValidator.validate_shots(1000) is True
        assert QuantumCircuitValidator.validate_shots(1) is True
        
        # Invalid shots
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_shots(0)
        
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_shots(-100)
        
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_shots(2_000_000)
    
    def test_validate_gate_set(self):
        """Test gate set validation."""
        # Valid gates
        valid_gates = ['h', 'x', 'cx', 'measure']
        assert QuantumCircuitValidator.validate_gate_set(valid_gates) is True
        
        # Valid with custom allowed set
        custom_allowed = ['h', 'x', 'y', 'z']
        assert QuantumCircuitValidator.validate_gate_set(['h', 'x'], custom_allowed) is True
        
        # Invalid gate
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_gate_set(['invalid_gate'])
        
        # Gate not in custom allowed set
        with pytest.raises(CircuitValidationError):
            QuantumCircuitValidator.validate_gate_set(['cx'], ['h', 'x'])


class TestInputSanitizer:
    """Test input sanitization functions."""
    
    def test_sanitize_identifier(self):
        """Test identifier sanitization."""
        # Valid identifier
        assert InputSanitizer.sanitize_identifier("test_circuit") == "test_circuit"
        
        # Contains invalid characters
        assert InputSanitizer.sanitize_identifier("test-circuit!") == "test_circuit_"
        
        # Starts with number
        assert InputSanitizer.sanitize_identifier("123test") == "_123test"
        
        # Too long
        long_name = "a" * 150
        result = InputSanitizer.sanitize_identifier(long_name)
        assert len(result) == 100
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Valid filename
        assert InputSanitizer.sanitize_filename("test_circuit.py") == "test_circuit.py"
        
        # Contains dangerous characters
        assert InputSanitizer.sanitize_filename("test<>circuit.py") == "test__circuit.py"
        
        # Reserved name
        assert InputSanitizer.sanitize_filename("CON") == "CON_safe"
        
        # Empty filename
        assert InputSanitizer.sanitize_filename("") == "unnamed"
    
    def test_sanitize_numeric_input(self):
        """Test numeric input sanitization."""
        # Valid integers
        assert InputSanitizer.sanitize_numeric_input("42", integer_only=True) == 42
        assert InputSanitizer.sanitize_numeric_input(42.0, integer_only=True) == 42
        
        # Valid floats
        assert InputSanitizer.sanitize_numeric_input("3.14") == 3.14
        assert InputSanitizer.sanitize_numeric_input(2.718) == 2.718
        
        # With bounds
        assert InputSanitizer.sanitize_numeric_input("5", min_val=0, max_val=10) == 5
        
        # Outside bounds
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_numeric_input("15", min_val=0, max_val=10)
        
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_numeric_input("-5", min_val=0, max_val=10)
        
        # Invalid input
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_numeric_input("not_a_number")


if __name__ == '__main__':
    pytest.main([__file__])