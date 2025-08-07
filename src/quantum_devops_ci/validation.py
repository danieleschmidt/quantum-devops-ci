"""
Input validation and sanitization for quantum DevOps CI system.

This module provides comprehensive validation for user inputs, configuration,
and quantum circuit data to ensure system security and reliability.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .exceptions import (
    ConfigurationError, 
    CircuitValidationError, 
    SecurityError
)


class SecurityValidator:
    """Security-focused input validation."""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'open\s*\(',
        r'subprocess',
        r'os\.(system|popen|spawn)',
        r'importlib',
        r'\.\.__.*__',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'getattr',
        r'setattr',
        r'delattr',
        r'hasattr',
    ]
    
    # Safe quantum-related imports
    SAFE_IMPORTS = {
        'qiskit', 'cirq', 'pennylane', 'braket',
        'numpy', 'scipy', 'matplotlib', 'pandas',
        'json', 'yaml', 'datetime', 'typing', 'dataclasses'
    }
    
    @classmethod
    def validate_code_string(cls, code: str) -> bool:
        """
        Validate that code string is safe to execute.
        
        Args:
            code: Code string to validate
            
        Returns:
            True if safe, False otherwise
            
        Raises:
            SecurityError: If dangerous patterns found
        """
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        return True
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> bool:
        """
        Validate file path for security issues.
        
        Args:
            file_path: File path to validate
            
        Returns:
            True if safe
            
        Raises:
            SecurityError: If path is unsafe
        """
        # Check for path traversal attacks
        if '..' in file_path:
            raise SecurityError("Path traversal detected")
        
        # Check for absolute paths to sensitive areas
        sensitive_paths = ['/etc', '/bin', '/usr/bin', '/root', '/home']
        path_obj = Path(file_path)
        
        try:
            resolved_path = path_obj.resolve()
            for sensitive in sensitive_paths:
                if str(resolved_path).startswith(sensitive):
                    raise SecurityError(f"Access to sensitive path blocked: {sensitive}")
        except (OSError, RuntimeError):
            raise SecurityError("Invalid file path")
        
        return True
    
    @classmethod
    def sanitize_string(cls, input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            input_str: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # Limit length
        input_str = input_str[:max_length]
        
        # Remove null bytes and control characters
        input_str = ''.join(char for char in input_str 
                           if ord(char) >= 32 or char in '\t\n\r')
        
        return input_str
    
    @classmethod
    def validate_json_input(cls, json_data: Union[str, Dict]) -> Dict[str, Any]:
        """
        Validate and parse JSON input.
        
        Args:
            json_data: JSON string or dict
            
        Returns:
            Parsed JSON data
            
        Raises:
            SecurityError: If JSON is invalid or unsafe
        """
        if isinstance(json_data, str):
            # Limit JSON size
            if len(json_data) > 1_000_000:  # 1MB limit
                raise SecurityError("JSON input too large")
            
            try:
                parsed_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise SecurityError(f"Invalid JSON: {e}")
        else:
            parsed_data = json_data
        
        # Recursively validate JSON structure
        cls._validate_json_structure(parsed_data)
        
        return parsed_data
    
    @classmethod
    def _validate_json_structure(cls, data: Any, depth: int = 0) -> None:
        """Recursively validate JSON structure for security."""
        if depth > 100:  # Prevent deep nesting attacks
            raise SecurityError("JSON nesting too deep")
        
        if isinstance(data, dict):
            if len(data) > 1000:  # Limit dict size
                raise SecurityError("JSON object too large")
            
            for key, value in data.items():
                if not isinstance(key, str):
                    raise SecurityError("Non-string keys not allowed")
                
                if len(key) > 1000:
                    raise SecurityError("JSON key too long")
                
                cls._validate_json_structure(value, depth + 1)
        
        elif isinstance(data, list):
            if len(data) > 10000:  # Limit array size
                raise SecurityError("JSON array too large")
            
            for item in data:
                cls._validate_json_structure(item, depth + 1)
        
        elif isinstance(data, str):
            if len(data) > 100000:  # Limit string size
                raise SecurityError("JSON string too long")


class ConfigValidator:
    """Configuration validation."""
    
    REQUIRED_FIELDS = {
        'testing_config': ['noise_model', 'backend_preferences', 'tolerance'],
        'scheduling_config': ['optimization_goal', 'max_queue_time'],
        'monitoring_config': ['project_name', 'storage_path'],
        'cost_config': ['monthly_budget', 'providers']
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration
            
        Returns:
            True if valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Check required fields
        required = cls.REQUIRED_FIELDS.get(config_type, [])
        for field in required:
            if field not in config:
                raise ConfigurationError(f"Missing required field: {field}")
        
        # Type-specific validation
        if config_type == 'testing_config':
            cls._validate_testing_config(config)
        elif config_type == 'scheduling_config':
            cls._validate_scheduling_config(config)
        elif config_type == 'monitoring_config':
            cls._validate_monitoring_config(config)
        elif config_type == 'cost_config':
            cls._validate_cost_config(config)
        
        return True
    
    @classmethod
    def _validate_testing_config(cls, config: Dict[str, Any]) -> None:
        """Validate testing configuration."""
        # Validate noise model
        if not isinstance(config.get('noise_model'), dict):
            raise ConfigurationError("noise_model must be a dictionary")
        
        # Validate tolerance
        tolerance = config.get('tolerance')
        if not isinstance(tolerance, (int, float)) or tolerance < 0 or tolerance > 1:
            raise ConfigurationError("tolerance must be a number between 0 and 1")
        
        # Validate backend preferences
        backend_prefs = config.get('backend_preferences')
        if not isinstance(backend_prefs, list):
            raise ConfigurationError("backend_preferences must be a list")
    
    @classmethod
    def _validate_scheduling_config(cls, config: Dict[str, Any]) -> None:
        """Validate scheduling configuration."""
        # Validate optimization goal
        valid_goals = ['minimize_cost', 'minimize_time', 'maximize_throughput']
        goal = config.get('optimization_goal')
        if goal not in valid_goals:
            raise ConfigurationError(f"optimization_goal must be one of: {valid_goals}")
        
        # Validate max queue time
        max_time = config.get('max_queue_time')
        if not isinstance(max_time, (int, float)) or max_time <= 0:
            raise ConfigurationError("max_queue_time must be a positive number")
    
    @classmethod
    def _validate_monitoring_config(cls, config: Dict[str, Any]) -> None:
        """Validate monitoring configuration."""
        # Validate project name
        project_name = config.get('project_name')
        if not isinstance(project_name, str) or len(project_name) == 0:
            raise ConfigurationError("project_name must be a non-empty string")
        
        # Validate storage path
        storage_path = config.get('storage_path')
        if storage_path is not None:
            SecurityValidator.validate_file_path(str(storage_path))
    
    @classmethod
    def _validate_cost_config(cls, config: Dict[str, Any]) -> None:
        """Validate cost configuration."""
        # Validate monthly budget
        budget = config.get('monthly_budget')
        if not isinstance(budget, (int, float)) or budget <= 0:
            raise ConfigurationError("monthly_budget must be a positive number")
        
        # Validate providers
        providers = config.get('providers')
        if not isinstance(providers, list) or len(providers) == 0:
            raise ConfigurationError("providers must be a non-empty list")


class QuantumCircuitValidator:
    """Quantum circuit validation."""
    
    MAX_QUBITS = 100
    MAX_GATES = 10000
    MAX_DEPTH = 1000
    
    @classmethod
    def validate_circuit_parameters(cls, 
                                   num_qubits: int, 
                                   gate_count: int, 
                                   depth: int) -> bool:
        """
        Validate basic circuit parameters.
        
        Args:
            num_qubits: Number of qubits
            gate_count: Number of gates
            depth: Circuit depth
            
        Returns:
            True if valid
            
        Raises:
            CircuitValidationError: If parameters are invalid
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise CircuitValidationError("Number of qubits must be a positive integer")
        
        if num_qubits > cls.MAX_QUBITS:
            raise CircuitValidationError(f"Too many qubits: {num_qubits} > {cls.MAX_QUBITS}")
        
        if not isinstance(gate_count, int) or gate_count < 0:
            raise CircuitValidationError("Gate count must be a non-negative integer")
        
        if gate_count > cls.MAX_GATES:
            raise CircuitValidationError(f"Too many gates: {gate_count} > {cls.MAX_GATES}")
        
        if not isinstance(depth, int) or depth < 0:
            raise CircuitValidationError("Circuit depth must be a non-negative integer")
        
        if depth > cls.MAX_DEPTH:
            raise CircuitValidationError(f"Circuit too deep: {depth} > {cls.MAX_DEPTH}")
        
        return True
    
    @classmethod
    def validate_shots(cls, shots: int) -> bool:
        """
        Validate number of shots.
        
        Args:
            shots: Number of shots
            
        Returns:
            True if valid
            
        Raises:
            CircuitValidationError: If shots is invalid
        """
        if not isinstance(shots, int) or shots <= 0:
            raise CircuitValidationError("Shots must be a positive integer")
        
        if shots > 1_000_000:  # 1M shot limit
            raise CircuitValidationError(f"Too many shots: {shots} > 1,000,000")
        
        return True
    
    @classmethod
    def validate_gate_set(cls, gates: List[str], allowed_gates: Optional[List[str]] = None) -> bool:
        """
        Validate circuit gate set.
        
        Args:
            gates: List of gate names used in circuit
            allowed_gates: Optional list of allowed gates
            
        Returns:
            True if valid
            
        Raises:
            CircuitValidationError: If gates are invalid
        """
        if not isinstance(gates, list):
            raise CircuitValidationError("Gates must be a list")
        
        # Default allowed gates for common quantum frameworks
        if allowed_gates is None:
            allowed_gates = [
                'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg',
                'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
                'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
                'ccx', 'cswap', 'swap', 'iswap',
                'barrier', 'measure', 'reset'
            ]
        
        for gate in gates:
            if gate not in allowed_gates:
                raise CircuitValidationError(f"Gate '{gate}' not in allowed set")
        
        return True


class InputSanitizer:
    """Input sanitization utilities."""
    
    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """
        Sanitize identifier names (variable names, etc.).
        
        Args:
            name: Identifier to sanitize
            
        Returns:
            Sanitized identifier
        """
        # Keep only alphanumeric characters and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it starts with letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f'_{sanitized}'
        
        # Limit length
        return sanitized[:100]
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe filesystem usage.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        filename = filename[:255]
        
        # Avoid reserved names on Windows
        reserved = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                   'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                   'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        
        if filename.upper() in reserved:
            filename = f'{filename}_safe'
        
        return filename or 'unnamed'
    
    @staticmethod
    def sanitize_numeric_input(value: Any, 
                              min_val: Optional[float] = None,
                              max_val: Optional[float] = None,
                              integer_only: bool = False) -> Union[int, float]:
        """
        Sanitize and validate numeric input.
        
        Args:
            value: Input value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            integer_only: Whether to enforce integer values
            
        Returns:
            Sanitized numeric value
            
        Raises:
            ValueError: If value is invalid
        """
        # Convert to number
        try:
            if integer_only:
                num_val = int(value)
            else:
                num_val = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")
        
        # Check bounds
        if min_val is not None and num_val < min_val:
            raise ValueError(f"Value too small: {num_val} < {min_val}")
        
        if max_val is not None and num_val > max_val:
            raise ValueError(f"Value too large: {num_val} > {max_val}")
        
        return num_val


# Validation decorators
def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Validation functions for each parameter
        
    Example:
        @validate_inputs(shots=lambda x: QuantumCircuitValidator.validate_shots(x))
        def run_circuit(shots):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validator(value)
                    except Exception as e:
                        raise ConfigurationError(f"Validation failed for {param_name}: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_safe_path(path_param: str):
    """
    Decorator to validate file paths are safe.
    
    Args:
        path_param: Name of the path parameter to validate
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            if path_param in bound_args.arguments:
                path_value = bound_args.arguments[path_param]
                if path_value is not None:
                    SecurityValidator.validate_file_path(str(path_value))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator