"""
Security utilities for quantum DevOps CI system.

This module provides security features including authentication,
authorization, secrets management, and audit logging.
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from .exceptions import AuthenticationError, AuthorizationError, SecurityError


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    permissions: List[str]
    session_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if security context is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions or 'admin' in self.permissions


@dataclass
class AuditEvent:
    """Audit log event."""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SecretsManager:
    """Manage sensitive configuration and API keys."""
    
    def __init__(self, secrets_file: Optional[str] = None):
        """Initialize secrets manager."""
        self.secrets_file = Path(secrets_file) if secrets_file else Path.home() / '.quantum_devops_ci' / 'secrets.json'
        self.secrets = {}
        self.encryption_key = self._get_or_create_key()
        
        if self.secrets_file.exists():
            self._load_secrets()
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_file = self.secrets_file.parent / '.key'
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            key = secrets.token_bytes(32)  # 256-bit key
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            key_file.chmod(0o600)
            return key
    
    def _load_secrets(self):
        """Load encrypted secrets from file."""
        try:
            with open(self.secrets_file, 'r') as f:
                encrypted_data = json.load(f)
            
            # For now, store as plaintext (in production, implement proper encryption)
            self.secrets = encrypted_data
            
        except Exception as e:
            logging.warning(f"Failed to load secrets: {e}")
    
    def _save_secrets(self):
        """Save secrets to encrypted file."""
        try:
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            
            # For now, store as plaintext (in production, implement proper encryption)
            with open(self.secrets_file, 'w') as f:
                json.dump(self.secrets, f, indent=2)
            
            # Set restrictive permissions
            self.secrets_file.chmod(0o600)
            
        except Exception as e:
            logging.error(f"Failed to save secrets: {e}")
            raise SecurityError(f"Cannot save secrets: {e}")
    
    def set_secret(self, key: str, value: str):
        """Store a secret value."""
        self.secrets[key] = {
            'value': value,
            'created_at': datetime.now().isoformat(),
            'accessed_count': 0
        }
        self._save_secrets()
    
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret value."""
        if key not in self.secrets:
            return None
        
        secret_info = self.secrets[key]
        secret_info['accessed_count'] += 1
        secret_info['last_accessed'] = datetime.now().isoformat()
        
        self._save_secrets()
        return secret_info['value']
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        if key in self.secrets:
            del self.secrets[key]
            self._save_secrets()
            return True
        return False
    
    def list_secret_keys(self) -> List[str]:
        """List available secret keys (not values)."""
        return list(self.secrets.keys())


class AuthenticationManager:
    """Manage user authentication."""
    
    def __init__(self, secrets_manager: SecretsManager):
        self.secrets_manager = secrets_manager
        self.active_sessions = {}
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def create_session_token(self) -> str:
        """Create a secure session token."""
        return secrets.token_urlsafe(32)
    
    def authenticate_user(self, user_id: str, password: str) -> Optional[SecurityContext]:
        """
        Authenticate user with password.
        
        Args:
            user_id: User identifier
            password: User password
            
        Returns:
            Security context if authentication successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Check for lockout
        if self._is_user_locked_out(user_id):
            raise AuthenticationError("Account temporarily locked due to failed attempts")
        
        # Get stored password hash
        stored_hash = self.secrets_manager.get_secret(f"user_password_{user_id}")
        if stored_hash is None:
            self._record_failed_attempt(user_id)
            raise AuthenticationError("Invalid credentials")
        
        # Verify password
        if not self._verify_password(password, stored_hash):
            self._record_failed_attempt(user_id)
            raise AuthenticationError("Invalid credentials")
        
        # Clear failed attempts on successful authentication
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        
        # Create security context
        session_token = self.create_session_token()
        expires_at = datetime.now() + timedelta(hours=8)  # 8-hour session
        
        # Get user permissions
        permissions = self._get_user_permissions(user_id)
        
        context = SecurityContext(
            user_id=user_id,
            permissions=permissions,
            session_token=session_token,
            expires_at=expires_at
        )
        
        self.active_sessions[session_token] = context
        return context
    
    def authenticate_token(self, token: str) -> Optional[SecurityContext]:
        """
        Authenticate using session token.
        
        Args:
            token: Session token
            
        Returns:
            Security context if valid
        """
        if token not in self.active_sessions:
            return None
        
        context = self.active_sessions[token]
        
        if context.is_expired():
            del self.active_sessions[token]
            return None
        
        return context
    
    def create_user(self, user_id: str, password: str, permissions: List[str]):
        """Create a new user account."""
        # Hash password
        password_hash = self._hash_password(password)
        
        # Store password hash
        self.secrets_manager.set_secret(f"user_password_{user_id}", password_hash)
        
        # Store permissions
        self.secrets_manager.set_secret(f"user_permissions_{user_id}", json.dumps(permissions))
    
    def logout(self, token: str):
        """Logout and invalidate session."""
        if token in self.active_sessions:
            del self.active_sessions[token]
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = secrets.token_hex(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return f"{salt}:{pwdhash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, pwdhash = stored_hash.split(':')
            return hmac.compare_digest(
                pwdhash,
                hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
            )
        except ValueError:
            return False
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions."""
        permissions_json = self.secrets_manager.get_secret(f"user_permissions_{user_id}")
        if permissions_json:
            return json.loads(permissions_json)
        return ['user']  # Default permission
    
    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if user_id not in self.failed_attempts:
            return False
        
        attempts_info = self.failed_attempts[user_id]
        
        if attempts_info['count'] >= self.max_failed_attempts:
            if datetime.now() - attempts_info['last_attempt'] < self.lockout_duration:
                return True
            else:
                # Lockout expired, clear attempts
                del self.failed_attempts[user_id]
        
        return False
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt."""
        now = datetime.now()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = {'count': 0, 'last_attempt': now}
        
        attempts_info = self.failed_attempts[user_id]
        
        # Reset count if last attempt was more than lockout duration ago
        if now - attempts_info['last_attempt'] > self.lockout_duration:
            attempts_info['count'] = 0
        
        attempts_info['count'] += 1
        attempts_info['last_attempt'] = now


class AuthorizationManager:
    """Manage access control and permissions."""
    
    # Define available permissions
    PERMISSIONS = {
        'admin': 'Full system access',
        'user': 'Basic user access',
        'circuit.read': 'Read quantum circuits',
        'circuit.write': 'Create/modify quantum circuits',
        'circuit.execute': 'Execute quantum circuits',
        'test.read': 'Read test results',
        'test.write': 'Create/modify tests',
        'test.execute': 'Execute tests',
        'schedule.read': 'Read job schedules',
        'schedule.write': 'Create/modify job schedules',
        'monitor.read': 'Read monitoring data',
        'monitor.write': 'Configure monitoring',
        'cost.read': 'Read cost data',
        'cost.write': 'Configure cost settings',
        'config.read': 'Read configuration',
        'config.write': 'Modify configuration'
    }
    
    @classmethod
    def check_permission(cls, context: SecurityContext, required_permission: str):
        """
        Check if security context has required permission.
        
        Args:
            context: Security context
            required_permission: Required permission
            
        Raises:
            AuthorizationError: If permission denied
        """
        if context.is_expired():
            raise AuthorizationError("Security context expired")
        
        if not context.has_permission(required_permission):
            raise AuthorizationError(f"Permission denied: {required_permission}")
    
    @classmethod
    def get_available_permissions(cls) -> Dict[str, str]:
        """Get list of available permissions."""
        return cls.PERMISSIONS.copy()


class AuditLogger:
    """Audit logging for security events."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger."""
        self.log_file = Path(log_file) if log_file else Path.home() / '.quantum_devops_ci' / 'audit.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('quantum_audit')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
    
    def log_event(self, event: AuditEvent):
        """Log audit event."""
        event_data = {
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'action': event.action,
            'resource': event.resource,
            'result': event.result,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'details': event.details
        }
        
        self.logger.info(json.dumps(event_data))
    
    def log_authentication(self, user_id: str, success: bool, ip_address: Optional[str] = None):
        """Log authentication attempt."""
        event = AuditEvent(
            timestamp=datetime.now(),
            user_id=user_id,
            action='authenticate',
            resource='system',
            result='success' if success else 'failure',
            ip_address=ip_address
        )
        self.log_event(event)
    
    def log_authorization(self, user_id: str, permission: str, resource: str, granted: bool):
        """Log authorization check."""
        event = AuditEvent(
            timestamp=datetime.now(),
            user_id=user_id,
            action='authorize',
            resource=resource,
            result='granted' if granted else 'denied',
            details={'permission': permission}
        )
        self.log_event(event)
    
    def log_resource_access(self, user_id: str, action: str, resource: str, success: bool):
        """Log resource access."""
        event = AuditEvent(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            result='success' if success else 'failure'
        )
        self.log_event(event)


class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, 
                 secrets_file: Optional[str] = None,
                 audit_log_file: Optional[str] = None):
        """Initialize security manager."""
        self.secrets_manager = SecretsManager(secrets_file)
        self.auth_manager = AuthenticationManager(self.secrets_manager)
        self.audit_logger = AuditLogger(audit_log_file)
    
    def authenticate(self, user_id: str, password: str, ip_address: Optional[str] = None) -> SecurityContext:
        """
        Authenticate user and log attempt.
        
        Args:
            user_id: User identifier
            password: User password
            ip_address: Client IP address
            
        Returns:
            Security context
        """
        try:
            context = self.auth_manager.authenticate_user(user_id, password)
            self.audit_logger.log_authentication(user_id, True, ip_address)
            return context
        except AuthenticationError:
            self.audit_logger.log_authentication(user_id, False, ip_address)
            raise
    
    def authorize(self, context: SecurityContext, permission: str, resource: str):
        """
        Check authorization and log attempt.
        
        Args:
            context: Security context
            permission: Required permission
            resource: Resource being accessed
        """
        try:
            AuthorizationManager.check_permission(context, permission)
            self.audit_logger.log_authorization(context.user_id, permission, resource, True)
        except AuthorizationError:
            self.audit_logger.log_authorization(context.user_id, permission, resource, False)
            raise
    
    def create_default_admin(self, admin_password: str):
        """Create default admin user if none exists."""
        admin_hash = self.secrets_manager.get_secret("user_password_admin")
        if admin_hash is None:
            self.auth_manager.create_user("admin", admin_password, ["admin"])
            logging.info("Default admin user created")


# Security decorators
def requires_auth(permission: str):
    """
    Decorator to require authentication and authorization.
    
    Args:
        permission: Required permission
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Look for security_context in kwargs
            security_context = kwargs.get('security_context')
            if security_context is None:
                raise AuthenticationError("Authentication required")
            
            # Check authorization
            AuthorizationManager.check_permission(security_context, permission)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def audit_action(action: str, resource: str):
    """
    Decorator to audit function calls.
    
    Args:
        action: Action being performed
        resource: Resource being accessed
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_context = kwargs.get('security_context')
            audit_logger = kwargs.get('audit_logger')
            
            if security_context and audit_logger:
                try:
                    result = func(*args, **kwargs)
                    audit_logger.log_resource_access(
                        security_context.user_id, action, resource, True
                    )
                    return result
                except Exception:
                    audit_logger.log_resource_access(
                        security_context.user_id, action, resource, False
                    )
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator