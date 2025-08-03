"""
Quantum DevOps CI/CD API Application.

This module provides the main Flask application for the quantum DevOps API.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    jsonify = None
    request = None
    CORS = None

from ..database.connection import get_connection
from ..monitoring import QuantumCIMonitor
from ..cost import CostOptimizer
from ..scheduling import QuantumJobScheduler
from ..testing import NoiseAwareTest
from ..deployment import QuantumDeployment, DeploymentConfig
from .middleware import (
    ValidationMiddleware,
    AuthenticationMiddleware, 
    RateLimitMiddleware,
    ErrorHandlerMiddleware
)
from .routes import (
    jobs_bp,
    cost_bp,
    monitoring_bp,
    testing_bp,
    deployment_bp
)


class QuantumDevOpsAPI:
    """Main API class for quantum DevOps CI/CD operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the API with configuration."""
        self.config = config or {}
        self.app = None
        self.monitor = None
        self.cost_optimizer = None
        self.job_scheduler = None
        self.tester = None
        self.deployment = None
        
        if FLASK_AVAILABLE:
            self._setup_services()
    
    def _setup_services(self):
        """Initialize quantum DevOps services."""
        # Initialize services
        self.monitor = QuantumCIMonitor(
            project=self.config.get('project_name', 'quantum-devops'),
            local_storage=self.config.get('local_storage', True)
        )
        
        self.cost_optimizer = CostOptimizer(
            monthly_budget=self.config.get('monthly_budget', 1000.0)
        )
        
        self.job_scheduler = QuantumJobScheduler(
            optimization_goal=self.config.get('optimization_goal', 'minimize_cost')
        )
        
        self.tester = NoiseAwareTest(
            default_shots=self.config.get('default_shots', 1000),
            timeout_seconds=self.config.get('timeout_seconds', 300)
        )
        
        deployment_config = DeploymentConfig(
            **self.config.get('deployment', {})
        )
        self.deployment = QuantumDeployment(
            service_name=self.config.get('service_name', 'quantum-service'),
            config=deployment_config
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall API health status."""
        if not FLASK_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Flask not available',
                'timestamp': datetime.now().isoformat()
            }
        
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'monitoring': 'healthy' if self.monitor else 'unavailable',
                'cost_optimization': 'healthy' if self.cost_optimizer else 'unavailable',
                'job_scheduling': 'healthy' if self.job_scheduler else 'unavailable',
                'testing': 'healthy' if self.tester else 'unavailable',
                'deployment': 'healthy' if self.deployment else 'unavailable'
            },
            'version': '1.0.0'
        }
        
        # Check if any services are unavailable
        if any(service == 'unavailable' for service in status['services'].values()):
            status['status'] = 'degraded'
        
        return status
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and available endpoints."""
        return {
            'name': 'Quantum DevOps CI/CD API',
            'version': '1.0.0',
            'description': 'REST API for quantum DevOps operations',
            'endpoints': {
                'health': '/api/health',
                'jobs': '/api/jobs',
                'cost': '/api/cost',
                'monitoring': '/api/monitoring',
                'testing': '/api/testing',
                'deployment': '/api/deployment'
            },
            'documentation': '/api/docs',
            'flask_available': FLASK_AVAILABLE
        }


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application."""
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask is not available. Install with: pip install flask flask-cors")
    
    app = Flask(__name__)
    
    # Load configuration
    app_config = config or {}
    app.config.update({
        'SECRET_KEY': app_config.get('SECRET_KEY', os.urandom(24)),
        'DEBUG': app_config.get('DEBUG', False),
        'TESTING': app_config.get('TESTING', False)
    })
    
    # Enable CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": app_config.get('CORS_ORIGINS', ["*"]),
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Initialize API
    api = QuantumDevOpsAPI(app_config)
    app.quantum_api = api
    
    # Register middleware
    ErrorHandlerMiddleware.init_app(app)
    ValidationMiddleware.init_app(app)
    
    if app_config.get('ENABLE_AUTH', False):
        AuthenticationMiddleware.init_app(app)
    
    if app_config.get('ENABLE_RATE_LIMITING', False):
        RateLimitMiddleware.init_app(app)
    
    # Register blueprints
    app.register_blueprint(jobs_bp, url_prefix='/api/jobs')
    app.register_blueprint(cost_bp, url_prefix='/api/cost')
    app.register_blueprint(monitoring_bp, url_prefix='/api/monitoring')
    app.register_blueprint(testing_bp, url_prefix='/api/testing')
    app.register_blueprint(deployment_bp, url_prefix='/api/deployment')
    
    # Root endpoints
    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        return jsonify(api.get_health_status())
    
    @app.route('/api/info')
    def info():
        """API information endpoint."""
        return jsonify(api.get_api_info())
    
    @app.route('/api')
    def root():
        """Root API endpoint."""
        return jsonify({
            'message': 'Quantum DevOps CI/CD API',
            'version': '1.0.0',
            'health': '/api/health',
            'info': '/api/info',
            'documentation': '/api/docs'
        })
    
    @app.route('/api/docs')
    def docs():
        """API documentation endpoint."""
        return jsonify({
            'title': 'Quantum DevOps CI/CD API Documentation',
            'version': '1.0.0',
            'base_url': request.base_url.replace('/docs', ''),
            'endpoints': {
                'Jobs Management': {
                    'GET /api/jobs': 'List all jobs',
                    'POST /api/jobs': 'Create new job',
                    'GET /api/jobs/{id}': 'Get job details',
                    'POST /api/jobs/schedule': 'Schedule job optimization',
                    'POST /api/jobs/batch': 'Submit batch jobs'
                },
                'Cost Optimization': {
                    'POST /api/cost/optimize': 'Optimize experiment costs',
                    'GET /api/cost/forecast': 'Get cost forecast',
                    'GET /api/cost/budget': 'Get budget status',
                    'POST /api/cost/usage': 'Update usage tracking'
                },
                'Monitoring': {
                    'GET /api/monitoring/builds': 'Get build metrics',
                    'GET /api/monitoring/hardware': 'Get hardware usage',
                    'GET /api/monitoring/performance': 'Get performance metrics',
                    'GET /api/monitoring/alerts': 'Get active alerts',
                    'POST /api/monitoring/build': 'Record build data'
                },
                'Testing': {
                    'POST /api/testing/run': 'Run quantum tests',
                    'POST /api/testing/noise': 'Run noise-aware tests',
                    'POST /api/testing/fidelity': 'Calculate fidelity',
                    'GET /api/testing/results': 'Get test results'
                },
                'Deployment': {
                    'POST /api/deployment/deploy': 'Deploy quantum algorithm',
                    'POST /api/deployment/rollback': 'Rollback deployment',
                    'GET /api/deployment/status': 'Get deployment status',
                    'POST /api/deployment/ab-test': 'Run A/B test'
                }
            },
            'authentication': 'Bearer token (if enabled)',
            'rate_limiting': 'Configurable per endpoint',
            'error_format': {
                'error': 'Error message',
                'code': 'Error code',
                'details': 'Additional details'
            }
        })
    
    return app


def run_development_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Run the development server."""
    if not FLASK_AVAILABLE:
        print("❌ Flask is not available. Install with: pip install flask flask-cors")
        return
    
    config = {
        'DEBUG': debug,
        'project_name': 'quantum-devops-dev',
        'monthly_budget': 500.0,
        'default_shots': 1000
    }
    
    app = create_app(config)
    
    print(f"🚀 Starting Quantum DevOps API server...")
    print(f"📡 Server running at http://{host}:{port}")
    print(f"📋 API documentation: http://{host}:{port}/api/docs")
    print(f"💚 Health check: http://{host}:{port}/api/health")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_development_server()
