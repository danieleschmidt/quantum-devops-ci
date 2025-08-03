"""
Middleware for quantum DevOps API.

Provides request validation, authentication, rate limiting,
and error handling middleware.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from functools import wraps

try:
    from flask import Flask, request, jsonify, g
    from werkzeug.exceptions import HTTPException
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    request = None
    jsonify = None
    g = None
    HTTPException = Exception


class ValidationMiddleware:
    """Request validation middleware."""
    
    @staticmethod
    def init_app(app: Flask):
        """Initialize validation middleware."""
        if not FLASK_AVAILABLE:
            return
        
        @app.before_request
        def validate_request():
            """Validate incoming requests."""
            # Skip validation for OPTIONS requests
            if request.method == 'OPTIONS':
                return
            
            # Validate Content-Type for POST/PUT requests
            if request.method in ['POST', 'PUT']:
                if not request.is_json and request.content_length > 0:
                    return jsonify({
                        'error': 'Invalid Content-Type',
                        'code': 'INVALID_CONTENT_TYPE',
                        'details': 'Content-Type must be application/json'
                    }), 400
            
            # Validate JSON payload
            if request.is_json:
                try:
                    request.get_json()
                except Exception as e:
                    return jsonify({
                        'error': 'Invalid JSON',
                        'code': 'INVALID_JSON',
                        'details': str(e)
                    }), 400
    
    @staticmethod
    def validate_schema(schema: Dict[str, Any]):
        """Decorator to validate request against schema."""
        def decorator(f: Callable):
            @wraps(f)
            def wrapper(*args, **kwargs):
                if not FLASK_AVAILABLE:
                    return f(*args, **kwargs)
                
                if request.method in ['POST', 'PUT']:
                    data = request.get_json() or {}
                    errors = ValidationMiddleware._validate_data(data, schema)
                    
                    if errors:
                        return jsonify({
                            'error': 'Validation failed',
                            'code': 'VALIDATION_ERROR',
                            'details': errors
                        }), 400
                
                return f(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def _validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> list:
        """Validate data against schema."""
        errors = []
        
        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get('properties', {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                field_type = field_schema.get('type')
                
                if field_type and not ValidationMiddleware._check_type(value, field_type):
                    errors.append(f"Invalid type for field {field}: expected {field_type}")
                
                # Check constraints
                if 'minimum' in field_schema and isinstance(value, (int, float)):
                    if value < field_schema['minimum']:
                        errors.append(f"Field {field} below minimum value {field_schema['minimum']}")
                
                if 'maximum' in field_schema and isinstance(value, (int, float)):
                    if value > field_schema['maximum']:
                        errors.append(f"Field {field} above maximum value {field_schema['maximum']}")
        
        return errors
    
    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True


class AuthenticationMiddleware:
    """Authentication middleware."""
    
    @staticmethod
    def init_app(app: Flask):
        """Initialize authentication middleware."""
        if not FLASK_AVAILABLE:
            return
        
        @app.before_request
        def authenticate_request():
            """Authenticate incoming requests."""
            # Skip authentication for health check and docs
            if request.endpoint in ['health', 'info', 'root', 'docs']:
                return
            
            # Skip authentication for OPTIONS requests
            if request.method == 'OPTIONS':
                return
            
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({
                    'error': 'Authentication required',
                    'code': 'MISSING_AUTH',
                    'details': 'Authorization header is required'
                }), 401
            
            # Parse Bearer token
            try:
                scheme, token = auth_header.split(' ', 1)
                if scheme.lower() != 'bearer':
                    raise ValueError("Invalid scheme")
            except ValueError:
                return jsonify({
                    'error': 'Invalid authorization header',
                    'code': 'INVALID_AUTH_HEADER',
                    'details': 'Authorization header must be in format: Bearer <token>'
                }), 401
            
            # Validate token (simplified validation)
            if not AuthenticationMiddleware._validate_token(token):
                return jsonify({
                    'error': 'Invalid token',
                    'code': 'INVALID_TOKEN',
                    'details': 'The provided token is invalid or expired'
                }), 401
            
            # Store user info in request context
            g.user = AuthenticationMiddleware._get_user_from_token(token)
    
    @staticmethod
    def _validate_token(token: str) -> bool:
        """Validate authentication token."""
        # Simple validation - in production, use proper JWT validation
        return len(token) >= 10 and token.isalnum()
    
    @staticmethod
    def _get_user_from_token(token: str) -> Dict[str, Any]:
        """Extract user information from token."""
        # Simplified user extraction - in production, decode JWT
        return {
            'id': 'user_' + token[:8],
            'permissions': ['read', 'write']
        }


class RateLimitMiddleware:
    """Rate limiting middleware."""
    
    def __init__(self):
        self.requests = {}  # In-memory storage (use Redis in production)
        self.limits = {
            'default': {'requests': 100, 'window': 60},  # 100 requests per minute
            'POST': {'requests': 20, 'window': 60},      # 20 POST requests per minute
            'PUT': {'requests': 20, 'window': 60},       # 20 PUT requests per minute
        }
    
    @staticmethod
    def init_app(app: Flask):
        """Initialize rate limiting middleware."""
        if not FLASK_AVAILABLE:
            return
        
        middleware = RateLimitMiddleware()
        
        @app.before_request
        def check_rate_limit():
            """Check rate limits for requests."""
            # Skip rate limiting for health check
            if request.endpoint in ['health', 'info', 'root', 'docs']:
                return
            
            client_ip = request.remote_addr
            method = request.method
            
            # Check rate limit
            if not middleware._check_rate_limit(client_ip, method):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'code': 'RATE_LIMIT_EXCEEDED',
                    'details': 'Too many requests. Please try again later.'
                }), 429
    
    def _check_rate_limit(self, client_ip: str, method: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        
        # Get rate limit for method or use default
        limit_config = self.limits.get(method, self.limits['default'])
        max_requests = limit_config['requests']
        window_seconds = limit_config['window']
        
        # Clean old requests
        self._clean_old_requests(current_time, window_seconds)
        
        # Get client's request history
        key = f"{client_ip}:{method}"
        if key not in self.requests:
            self.requests[key] = []
        
        client_requests = self.requests[key]
        
        # Check if within limit
        if len(client_requests) >= max_requests:
            return False
        
        # Add current request
        client_requests.append(current_time)
        return True
    
    def _clean_old_requests(self, current_time: float, window_seconds: int):
        """Remove old requests from tracking."""
        cutoff_time = current_time - window_seconds
        
        for key in list(self.requests.keys()):
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[key]:
                del self.requests[key]


class ErrorHandlerMiddleware:
    """Error handling middleware."""
    
    @staticmethod
    def init_app(app: Flask):
        """Initialize error handling middleware."""
        if not FLASK_AVAILABLE:
            return
        
        @app.errorhandler(Exception)
        def handle_exception(e):
            """Handle all exceptions."""
            # Handle HTTP exceptions
            if isinstance(e, HTTPException):
                return jsonify({
                    'error': e.description,
                    'code': e.name.upper().replace(' ', '_'),
                    'status_code': e.code
                }), e.code
            
            # Handle other exceptions
            app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            
            # Don't expose internal errors in production
            if app.config.get('DEBUG'):
                error_details = str(e)
            else:
                error_details = 'An internal error occurred'
            
            return jsonify({
                'error': 'Internal server error',
                'code': 'INTERNAL_SERVER_ERROR',
                'details': error_details
            }), 500
        
        @app.errorhandler(404)
        def handle_not_found(e):
            """Handle 404 errors."""
            return jsonify({
                'error': 'Endpoint not found',
                'code': 'NOT_FOUND',
                'details': f'The requested endpoint {request.path} was not found'
            }), 404
        
        @app.errorhandler(405)
        def handle_method_not_allowed(e):
            """Handle 405 errors."""
            return jsonify({
                'error': 'Method not allowed',
                'code': 'METHOD_NOT_ALLOWED',
                'details': f'Method {request.method} is not allowed for {request.path}'
            }), 405


def require_permissions(*required_perms):
    """Decorator to require specific permissions."""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not FLASK_AVAILABLE:
                return f(*args, **kwargs)
            
            # Check if user is authenticated
            if not hasattr(g, 'user'):
                return jsonify({
                    'error': 'Authentication required',
                    'code': 'MISSING_AUTH'
                }), 401
            
            # Check permissions
            user_perms = g.user.get('permissions', [])
            
            for perm in required_perms:
                if perm not in user_perms:
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'code': 'INSUFFICIENT_PERMISSIONS',
                        'details': f'Required permission: {perm}'
                    }), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def log_request():
    """Decorator to log API requests."""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not FLASK_AVAILABLE:
                return f(*args, **kwargs)
            
            start_time = time.time()
            
            # Log request
            print(f"[{datetime.now().isoformat()}] {request.method} {request.path} - Started")
            
            try:
                response = f(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful response
                status_code = getattr(response, 'status_code', 200)
                print(f"[{datetime.now().isoformat()}] {request.method} {request.path} - {status_code} ({duration:.3f}s)")
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"[{datetime.now().isoformat()}] {request.method} {request.path} - ERROR ({duration:.3f}s): {str(e)}")
                raise
            
        return wrapper
    return decorator
