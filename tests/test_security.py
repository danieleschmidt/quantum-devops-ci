"""
Test suite for security module.
"""

import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from quantum_devops_ci.security import (
    SecurityContext, SecretsManager, AuthenticationManager,
    AuthorizationManager, AuditLogger, SecurityManager
)
from quantum_devops_ci.exceptions import (
    AuthenticationError, AuthorizationError, SecurityError
)


class TestSecurityContext:
    """Test security context functionality."""
    
    def test_security_context_creation(self):
        """Test creating security context."""
        context = SecurityContext(
            user_id="test_user",
            permissions=["read", "write"],
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        assert context.user_id == "test_user"
        assert "read" in context.permissions
        assert not context.is_expired()
    
    def test_security_context_expiration(self):
        """Test security context expiration."""
        # Expired context
        context = SecurityContext(
            user_id="test_user",
            permissions=["read"],
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        assert context.is_expired()
    
    def test_has_permission(self):
        """Test permission checking."""
        context = SecurityContext(
            user_id="test_user",
            permissions=["read", "write"]
        )
        
        assert context.has_permission("read")
        assert context.has_permission("write")
        assert not context.has_permission("admin")
        
        # Admin permission grants all
        admin_context = SecurityContext(
            user_id="admin_user",
            permissions=["admin"]
        )
        
        assert admin_context.has_permission("read")
        assert admin_context.has_permission("write")
        assert admin_context.has_permission("delete")


class TestSecretsManager:
    """Test secrets management."""
    
    def test_secrets_manager_basic(self):
        """Test basic secrets operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            manager = SecretsManager(secrets_file)
            
            # Set and get secret
            manager.set_secret("api_key", "secret_value_123")
            retrieved = manager.get_secret("api_key")
            assert retrieved == "secret_value_123"
            
            # Get non-existent secret
            assert manager.get_secret("nonexistent") is None
            
            # List secret keys
            keys = manager.list_secret_keys()
            assert "api_key" in keys
    
    def test_secrets_persistence(self):
        """Test that secrets persist across instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            
            # First manager instance
            manager1 = SecretsManager(secrets_file)
            manager1.set_secret("persistent_key", "persistent_value")
            
            # Second manager instance
            manager2 = SecretsManager(secrets_file)
            retrieved = manager2.get_secret("persistent_key")
            assert retrieved == "persistent_value"
    
    def test_delete_secret(self):
        """Test secret deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            manager = SecretsManager(secrets_file)
            
            manager.set_secret("temp_secret", "temp_value")
            assert manager.get_secret("temp_secret") == "temp_value"
            
            # Delete secret
            assert manager.delete_secret("temp_secret") is True
            assert manager.get_secret("temp_secret") is None
            
            # Delete non-existent secret
            assert manager.delete_secret("nonexistent") is False


class TestAuthenticationManager:
    """Test authentication functionality."""
    
    def test_create_user_and_authenticate(self):
        """Test user creation and authentication."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            secrets_manager = SecretsManager(secrets_file)
            auth_manager = AuthenticationManager(secrets_manager)
            
            # Create user
            auth_manager.create_user("test_user", "test_password", ["read", "write"])
            
            # Authenticate successfully
            context = auth_manager.authenticate_user("test_user", "test_password")
            assert context.user_id == "test_user"
            assert "read" in context.permissions
            assert context.session_token is not None
    
    def test_authentication_failure(self):
        """Test authentication failure scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            secrets_manager = SecretsManager(secrets_file)
            auth_manager = AuthenticationManager(secrets_manager)
            
            # Try to authenticate non-existent user
            with pytest.raises(AuthenticationError):
                auth_manager.authenticate_user("nonexistent", "password")
            
            # Create user and try wrong password
            auth_manager.create_user("test_user", "correct_password", ["read"])
            
            with pytest.raises(AuthenticationError):
                auth_manager.authenticate_user("test_user", "wrong_password")
    
    def test_token_authentication(self):
        """Test token-based authentication."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            secrets_manager = SecretsManager(secrets_file)
            auth_manager = AuthenticationManager(secrets_manager)
            
            # Create and authenticate user
            auth_manager.create_user("test_user", "password", ["read"])
            context = auth_manager.authenticate_user("test_user", "password")
            
            # Authenticate with token
            token_context = auth_manager.authenticate_token(context.session_token)
            assert token_context.user_id == "test_user"
            
            # Invalid token
            invalid_context = auth_manager.authenticate_token("invalid_token")
            assert invalid_context is None
    
    def test_account_lockout(self):
        """Test account lockout after failed attempts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            secrets_manager = SecretsManager(secrets_file)
            auth_manager = AuthenticationManager(secrets_manager)
            auth_manager.max_failed_attempts = 3  # Lower for testing
            
            # Create user
            auth_manager.create_user("test_user", "correct_password", ["read"])
            
            # Make failed attempts
            for i in range(3):
                with pytest.raises(AuthenticationError):
                    auth_manager.authenticate_user("test_user", "wrong_password")
            
            # Account should now be locked
            with pytest.raises(AuthenticationError, match="locked"):
                auth_manager.authenticate_user("test_user", "wrong_password")


class TestAuthorizationManager:
    """Test authorization functionality."""
    
    def test_check_permission_success(self):
        """Test successful permission check."""
        context = SecurityContext(
            user_id="test_user",
            permissions=["read", "write"]
        )
        
        # Should not raise exception
        AuthorizationManager.check_permission(context, "read")
        AuthorizationManager.check_permission(context, "write")
    
    def test_check_permission_failure(self):
        """Test failed permission check."""
        context = SecurityContext(
            user_id="test_user",
            permissions=["read"]
        )
        
        # Should raise AuthorizationError
        with pytest.raises(AuthorizationError):
            AuthorizationManager.check_permission(context, "write")
    
    def test_check_permission_expired_context(self):
        """Test permission check with expired context."""
        context = SecurityContext(
            user_id="test_user",
            permissions=["read"],
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        with pytest.raises(AuthorizationError, match="expired"):
            AuthorizationManager.check_permission(context, "read")
    
    def test_admin_permission(self):
        """Test that admin permission grants all access."""
        admin_context = SecurityContext(
            user_id="admin_user",
            permissions=["admin"]
        )
        
        # Admin should have access to everything
        AuthorizationManager.check_permission(admin_context, "read")
        AuthorizationManager.check_permission(admin_context, "write")
        AuthorizationManager.check_permission(admin_context, "delete")
        AuthorizationManager.check_permission(admin_context, "config.write")


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def test_audit_logger_basic(self):
        """Test basic audit logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'audit.log')
            logger = AuditLogger(log_file)
            
            # Log authentication event
            logger.log_authentication("test_user", True, "192.168.1.100")
            
            # Check log file exists and contains data
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "test_user" in content
                assert "authenticate" in content
                assert "192.168.1.100" in content
    
    def test_audit_logger_authorization(self):
        """Test authorization audit logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'audit.log')
            logger = AuditLogger(log_file)
            
            # Log authorization events
            logger.log_authorization("test_user", "read", "circuit_data", True)
            logger.log_authorization("test_user", "write", "config", False)
            
            # Check log content
            with open(log_file, 'r') as f:
                content = f.read()
                assert "authorize" in content
                assert "circuit_data" in content
                assert "granted" in content
                assert "denied" in content
    
    def test_audit_logger_resource_access(self):
        """Test resource access audit logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'audit.log')
            logger = AuditLogger(log_file)
            
            # Log resource access
            logger.log_resource_access("test_user", "execute", "quantum_circuit", True)
            
            # Check log content
            with open(log_file, 'r') as f:
                content = f.read()
                assert "execute" in content
                assert "quantum_circuit" in content
                assert "success" in content


class TestSecurityManager:
    """Test integrated security manager."""
    
    def test_security_manager_integration(self):
        """Test security manager integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = os.path.join(temp_dir, 'secrets.json')
            audit_file = os.path.join(temp_dir, 'audit.log')
            
            security_manager = SecurityManager(secrets_file, audit_file)
            
            # Create default admin
            security_manager.create_default_admin("admin_password")
            
            # Authenticate admin
            context = security_manager.authenticate("admin", "admin_password")
            assert context.user_id == "admin"
            assert "admin" in context.permissions
            
            # Test authorization
            security_manager.authorize(context, "read", "test_resource")
            
            # Check that audit log was created
            assert os.path.exists(audit_file)


if __name__ == '__main__':
    pytest.main([__file__])