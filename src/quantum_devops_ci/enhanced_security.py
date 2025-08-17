"""
Enhanced security framework for quantum DevOps CI/CD.
Generation 2 implementation with comprehensive security controls.
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .exceptions import QuantumSecurityError, QuantumValidationError

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for quantum operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatLevel(Enum):
    """Threat assessment levels."""
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"

@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    timestamp: float
    action: str
    user: str
    resource: str
    threat_level: ThreatLevel
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

class QuantumSecurityManager:
    """
    Comprehensive security manager for quantum DevOps operations.
    Implements defense-in-depth security model.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.audit_log: List[SecurityAuditEntry] = []
        
        # Initialize encryption
        self._encryption_key = self._generate_encryption_key()
        self._cipher_suite = Fernet(self._encryption_key)
        
        # Security policies
        self._forbidden_patterns = self._load_forbidden_patterns()
        self._allowed_gates = self._load_allowed_gates()
        self._rate_limits = self._initialize_rate_limits()
        
        # Session management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._failed_attempts: Dict[str, List[float]] = {}
        
        # Threat detection
        self._threat_signatures = self._load_threat_signatures()
        
    def _generate_encryption_key(self) -> bytes:
        """Generate secure encryption key for sensitive data."""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _load_forbidden_patterns(self) -> List[str]:
        """Load patterns that indicate malicious quantum circuits."""
        return [
            r'while\s*\(',          # Infinite loops
            r'for\s*\(',            # Potential DoS loops
            r'eval\s*\(',           # Code injection
            r'exec\s*\(',           # Code execution
            r'import\s+os',         # System access
            r'import\s+subprocess', # Process execution
            r'__import__',          # Dynamic imports
            r'\.\./',               # Path traversal
            r'\/etc\/',             # System file access
            r'\/proc\/',            # Process information
            r'OPENQASM\s+[3-9]',    # Unsupported QASM versions
        ]
    
    def _load_allowed_gates(self) -> Set[str]:
        """Load whitelist of allowed quantum gates."""
        return {
            # Single qubit gates
            'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
            'id', 'phase',
            
            # Two qubit gates
            'cx', 'cy', 'cz', 'swap', 'iswap',
            'crx', 'cry', 'crz', 'cu1', 'cu3',
            
            # Multi-qubit gates
            'ccx', 'cswap', 'mcx', 'mcy', 'mcz',
            
            # Measurement
            'measure', 'measure_all',
            
            # Barriers and reset
            'barrier', 'reset'
        }
    
    def _initialize_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rate limiting configuration."""
        return {
            'circuit_execution': {
                'requests_per_minute': 60,
                'requests_per_hour': 1000,
                'max_circuit_depth': 200,
                'max_qubits': 50
            },
            'api_calls': {
                'requests_per_minute': 100,
                'requests_per_hour': 5000
            },
            'authentication': {
                'max_failed_attempts': 5,
                'lockout_duration_minutes': 15
            }
        }
    
    def _load_threat_signatures(self) -> List[Dict[str, Any]]:
        """Load threat detection signatures."""
        return [
            {
                'name': 'excessive_gate_count',
                'pattern': lambda circuit: self._check_excessive_gates(circuit),
                'severity': ThreatLevel.SUSPICIOUS,
                'description': 'Circuit contains excessive number of gates'
            },
            {
                'name': 'deep_circuit',
                'pattern': lambda circuit: self._check_circuit_depth(circuit),
                'severity': ThreatLevel.SUSPICIOUS,
                'description': 'Circuit depth exceeds safety limits'
            },
            {
                'name': 'forbidden_gate_pattern',
                'pattern': lambda circuit: self._check_forbidden_gates(circuit),
                'severity': ThreatLevel.MALICIOUS,
                'description': 'Circuit contains forbidden gate patterns'
            },
            {
                'name': 'resource_exhaustion',
                'pattern': lambda circuit: self._check_resource_usage(circuit),
                'severity': ThreatLevel.CRITICAL,
                'description': 'Circuit may cause resource exhaustion'
            }
        ]
    
    def validate_quantum_circuit(self, circuit: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Comprehensive quantum circuit security validation.
        
        Args:
            circuit: Quantum circuit to validate
            context: Optional context information (user, session, etc.)
            
        Returns:
            bool: True if circuit passes all security checks
            
        Raises:
            QuantumSecurityError: If circuit fails security validation
        """
        validation_context = context or {}
        threats_detected = []
        
        try:
            # Basic structure validation
            self._validate_circuit_structure(circuit)
            
            # Gate whitelist validation
            self._validate_allowed_gates(circuit)
            
            # Pattern-based threat detection
            for signature in self._threat_signatures:
                if signature['pattern'](circuit):
                    threat = {
                        'signature': signature['name'],
                        'severity': signature['severity'],
                        'description': signature['description']
                    }
                    threats_detected.append(threat)
                    
                    if signature['severity'] in [ThreatLevel.MALICIOUS, ThreatLevel.CRITICAL]:
                        self._log_security_event(
                            action="circuit_validation_failed",
                            threat_level=signature['severity'],
                            details={
                                'signature': signature['name'],
                                'description': signature['description'],
                                'circuit_info': self._extract_circuit_info(circuit)
                            },
                            **validation_context
                        )
                        raise QuantumSecurityError(f"Security threat detected: {signature['description']}")
            
            # Rate limiting check
            self._check_rate_limits(validation_context.get('user', 'unknown'), 'circuit_execution')
            
            # Log successful validation
            self._log_security_event(
                action="circuit_validation_passed",
                threat_level=ThreatLevel.BENIGN,
                details={
                    'threats_detected': len(threats_detected),
                    'circuit_info': self._extract_circuit_info(circuit)
                },
                **validation_context
            )
            
            return True
            
        except QuantumSecurityError:
            raise
        except Exception as e:
            logger.error(f"Circuit validation error: {e}")
            raise QuantumValidationError(f"Circuit validation failed: {e}")
    
    def _validate_circuit_structure(self, circuit: Any) -> None:
        """Validate basic circuit structure and properties."""
        if circuit is None:
            raise QuantumSecurityError("Circuit cannot be None")
        
        # Check for required attributes based on framework
        if hasattr(circuit, 'num_qubits'):
            if circuit.num_qubits <= 0:
                raise QuantumSecurityError("Circuit must have at least one qubit")
            
            if circuit.num_qubits > self._rate_limits['circuit_execution']['max_qubits']:
                raise QuantumSecurityError(f"Circuit exceeds maximum qubits limit: {circuit.num_qubits}")
        
        if hasattr(circuit, 'depth'):
            depth = circuit.depth()
            if depth > self._rate_limits['circuit_execution']['max_circuit_depth']:
                raise QuantumSecurityError(f"Circuit depth exceeds limit: {depth}")
    
    def _validate_allowed_gates(self, circuit: Any) -> None:
        """Validate that circuit only contains allowed gates."""
        try:
            # For Qiskit circuits
            if hasattr(circuit, 'count_ops'):
                ops = circuit.count_ops()
                for gate_name in ops.keys():
                    gate_name_lower = gate_name.lower()
                    if gate_name_lower not in self._allowed_gates:
                        raise QuantumSecurityError(f"Forbidden gate detected: {gate_name}")
            
            # For circuits with data attribute (Qiskit)
            elif hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    gate_name = instruction.operation.name.lower()
                    if gate_name not in self._allowed_gates:
                        raise QuantumSecurityError(f"Forbidden gate detected: {gate_name}")
            
            # Pattern-based validation for QASM
            if hasattr(circuit, 'qasm'):
                qasm_str = circuit.qasm()
                self._validate_qasm_security(qasm_str)
                
        except QuantumSecurityError:
            raise
        except Exception as e:
            logger.warning(f"Gate validation warning: {e}")
    
    def _validate_qasm_security(self, qasm_str: str) -> None:
        """Validate QASM code for security threats."""
        qasm_lower = qasm_str.lower()
        
        for pattern in self._forbidden_patterns:
            if re.search(pattern, qasm_lower):
                raise QuantumSecurityError(f"Forbidden pattern detected in QASM: {pattern}")
        
        # Check for excessive repetition (potential DoS)
        lines = qasm_str.split('\n')
        if len(lines) > 10000:
            raise QuantumSecurityError("QASM code too large, potential DoS attack")
        
        # Check for excessive register sizes
        register_pattern = r'qreg\s+\w+\[(\d+)\]'
        matches = re.findall(register_pattern, qasm_str)
        for match in matches:
            size = int(match)
            if size > self._rate_limits['circuit_execution']['max_qubits']:
                raise QuantumSecurityError(f"Register size too large: {size}")
    
    def _check_excessive_gates(self, circuit: Any) -> bool:
        """Check for excessive number of gates (potential DoS)."""
        try:
            if hasattr(circuit, 'size'):
                return circuit.size() > 10000
            elif hasattr(circuit, 'count_ops'):
                total_ops = sum(circuit.count_ops().values())
                return total_ops > 10000
            elif hasattr(circuit, '__len__'):
                return len(circuit) > 10000
        except:
            pass
        return False
    
    def _check_circuit_depth(self, circuit: Any) -> bool:
        """Check for excessive circuit depth."""
        try:
            if hasattr(circuit, 'depth'):
                return circuit.depth() > self._rate_limits['circuit_execution']['max_circuit_depth']
        except:
            pass
        return False
    
    def _check_forbidden_gates(self, circuit: Any) -> bool:
        """Check for forbidden gate patterns."""
        try:
            forbidden_gates = {'custom', 'unitary', 'unknown'}
            
            if hasattr(circuit, 'count_ops'):
                ops = circuit.count_ops()
                return any(gate.lower() in forbidden_gates for gate in ops.keys())
            
            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    gate_name = instruction.operation.name.lower()
                    if gate_name in forbidden_gates:
                        return True
        except:
            pass
        return False
    
    def _check_resource_usage(self, circuit: Any) -> bool:
        """Check for potential resource exhaustion."""
        try:
            # Check memory usage estimation
            if hasattr(circuit, 'num_qubits'):
                # Estimate memory usage for statevector simulation
                estimated_memory_mb = (2 ** circuit.num_qubits) * 16 / (1024 * 1024)
                if estimated_memory_mb > 1024:  # 1GB limit
                    return True
            
            # Check for potential infinite loops in custom gates
            if hasattr(circuit, 'qasm'):
                qasm_str = circuit.qasm()
                if 'while' in qasm_str.lower() or 'loop' in qasm_str.lower():
                    return True
                    
        except:
            pass
        return False
    
    def _extract_circuit_info(self, circuit: Any) -> Dict[str, Any]:
        """Extract safe circuit information for logging."""
        info = {}
        
        try:
            if hasattr(circuit, 'num_qubits'):
                info['num_qubits'] = circuit.num_qubits
            if hasattr(circuit, 'depth'):
                info['depth'] = circuit.depth()
            if hasattr(circuit, 'size'):
                info['size'] = circuit.size()
            if hasattr(circuit, 'count_ops'):
                info['gate_counts'] = dict(circuit.count_ops())
        except Exception as e:
            logger.warning(f"Failed to extract circuit info: {e}")
            
        return info
    
    def _check_rate_limits(self, user: str, operation: str) -> None:
        """Check rate limits for user operations."""
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_rate_limit_data(current_time)
        
        if user not in self._failed_attempts:
            self._failed_attempts[user] = []
        
        user_attempts = self._failed_attempts[user]
        
        # Check authentication lockout
        if operation == 'authentication':
            recent_failures = [
                attempt for attempt in user_attempts
                if current_time - attempt < 900  # 15 minutes
            ]
            
            if len(recent_failures) >= self._rate_limits['authentication']['max_failed_attempts']:
                raise QuantumSecurityError(f"User {user} is locked out due to too many failed attempts")
        
        # Check operation rate limits
        operation_limits = self._rate_limits.get(operation, {})
        
        # Check per-minute limit
        if 'requests_per_minute' in operation_limits:
            minute_attempts = [
                attempt for attempt in user_attempts
                if current_time - attempt < 60
            ]
            
            if len(minute_attempts) >= operation_limits['requests_per_minute']:
                raise QuantumSecurityError(f"Rate limit exceeded for {operation}: {len(minute_attempts)} requests per minute")
        
        # Record this attempt
        user_attempts.append(current_time)
    
    def _cleanup_rate_limit_data(self, current_time: float) -> None:
        """Clean up old rate limit data."""
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        for user in list(self._failed_attempts.keys()):
            self._failed_attempts[user] = [
                attempt for attempt in self._failed_attempts[user]
                if attempt > cutoff_time
            ]
            
            if not self._failed_attempts[user]:
                del self._failed_attempts[user]
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage."""
        try:
            encrypted_data = self._cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise QuantumSecurityError("Failed to encrypt sensitive data")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise QuantumSecurityError("Failed to decrypt sensitive data")
    
    def create_secure_session(self, user: str, **kwargs) -> str:
        """Create secure session for user."""
        session_id = secrets.token_urlsafe(32)
        current_time = time.time()
        
        session_data = {
            'user': user,
            'created': current_time,
            'last_activity': current_time,
            'ip_address': kwargs.get('ip_address'),
            'user_agent': kwargs.get('user_agent'),
            'permissions': kwargs.get('permissions', [])
        }
        
        self._active_sessions[session_id] = session_data
        
        self._log_security_event(
            action="session_created",
            user=user,
            threat_level=ThreatLevel.BENIGN,
            details={'session_id': session_id},
            session_id=session_id,
            ip_address=kwargs.get('ip_address')
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate user session."""
        if session_id not in self._active_sessions:
            return False
        
        session = self._active_sessions[session_id]
        current_time = time.time()
        
        # Check session timeout (24 hours)
        if current_time - session['created'] > 86400:
            del self._active_sessions[session_id]
            return False
        
        # Update last activity
        session['last_activity'] = current_time
        return True
    
    def _log_security_event(self, action: str, threat_level: ThreatLevel, 
                          details: Dict[str, Any], user: str = "unknown", 
                          resource: str = "", **kwargs) -> None:
        """Log security event to audit trail."""
        entry = SecurityAuditEntry(
            timestamp=time.time(),
            action=action,
            user=user,
            resource=resource,
            threat_level=threat_level,
            details=details,
            ip_address=kwargs.get('ip_address'),
            session_id=kwargs.get('session_id')
        )
        
        self.audit_log.append(entry)
        
        # Log to system logger based on threat level
        log_message = f"Security Event: {action} | User: {user} | Threat: {threat_level.value} | Details: {details}"
        
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical(log_message)
        elif threat_level == ThreatLevel.MALICIOUS:
            logger.error(log_message)
        elif threat_level == ThreatLevel.SUSPICIOUS:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for specified time period."""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_events = [
            entry for entry in self.audit_log
            if entry.timestamp > cutoff_time
        ]
        
        # Count events by type and threat level
        event_counts = {}
        threat_counts = {level: 0 for level in ThreatLevel}
        
        for event in recent_events:
            event_counts[event.action] = event_counts.get(event.action, 0) + 1
            threat_counts[event.threat_level] += 1
        
        # Identify top threats
        critical_events = [
            event for event in recent_events
            if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.MALICIOUS]
        ]
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_breakdown': event_counts,
            'threat_level_breakdown': {level.value: count for level, count in threat_counts.items()},
            'critical_events': [
                {
                    'timestamp': event.timestamp,
                    'action': event.action,
                    'user': event.user,
                    'threat_level': event.threat_level.value,
                    'details': event.details
                }
                for event in critical_events
            ],
            'active_sessions': len(self._active_sessions),
            'users_with_rate_limits': len(self._failed_attempts)
        }