"""
Database models for quantum DevOps CI/CD.

This module defines the data models used for persisting quantum CI/CD
metrics, build records, and other operational data.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod


@dataclass
class BaseModel(ABC):
    """Base model for all database entities."""
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data
    
    @classmethod
    @abstractmethod
    def table_name(cls) -> str:
        """Get the database table name for this model."""
        pass
    
    @classmethod
    @abstractmethod
    def create_table_sql(cls) -> str:
        """Get SQL statement to create table for this model."""
        pass


@dataclass
class BuildRecord(BaseModel):
    """Build record model."""
    commit_hash: str = ""
    branch: str = ""
    build_number: Optional[int] = None
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    circuit_count: int = 0
    total_gates: int = 0
    max_circuit_depth: int = 0
    estimated_fidelity: float = 0.0
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    noise_tests_passed: int = 0
    noise_tests_total: int = 0
    framework: str = ""
    backend: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "build_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS build_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_hash TEXT NOT NULL,
            branch TEXT NOT NULL,
            build_number INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds REAL,
            circuit_count INTEGER DEFAULT 0,
            total_gates INTEGER DEFAULT 0,
            max_circuit_depth INTEGER DEFAULT 0,
            estimated_fidelity REAL DEFAULT 0.0,
            tests_total INTEGER DEFAULT 0,
            tests_passed INTEGER DEFAULT 0,
            tests_failed INTEGER DEFAULT 0,
            noise_tests_passed INTEGER DEFAULT 0,
            noise_tests_total INTEGER DEFAULT 0,
            framework TEXT DEFAULT '',
            backend TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


@dataclass
class HardwareUsageRecord(BaseModel):
    """Hardware usage record model."""
    job_id: str = ""
    provider: str = ""
    backend: str = ""
    shots: int = 0
    circuit_depth: Optional[int] = None
    num_qubits: Optional[int] = None
    queue_time_minutes: float = 0.0
    execution_time_minutes: float = 0.0
    cost_usd: float = 0.0
    cost_currency: str = "USD"
    success: bool = True
    error_message: Optional[str] = None
    build_id: Optional[int] = None
    experiment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "hardware_usage_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS hardware_usage_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            backend TEXT NOT NULL,
            shots INTEGER DEFAULT 0,
            circuit_depth INTEGER,
            num_qubits INTEGER,
            queue_time_minutes REAL DEFAULT 0.0,
            execution_time_minutes REAL DEFAULT 0.0,
            cost_usd REAL DEFAULT 0.0,
            cost_currency TEXT DEFAULT 'USD',
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            build_id INTEGER,
            experiment_id TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


@dataclass
class TestResult(BaseModel):
    """Test result model."""
    test_name: str = ""
    test_class: str = ""
    test_file: str = ""
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    framework: str = ""
    backend: str = ""
    shots: int = 0
    fidelity: Optional[float] = None
    error_rate: Optional[float] = None
    measurement_counts: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    build_id: Optional[int] = None
    noise_model: Optional[str] = None
    optimization_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "test_results"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_name TEXT NOT NULL,
            test_class TEXT DEFAULT '',
            test_file TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds REAL DEFAULT 0.0,
            framework TEXT DEFAULT '',
            backend TEXT DEFAULT '',
            shots INTEGER DEFAULT 0,
            fidelity REAL,
            error_rate REAL,
            measurement_counts TEXT DEFAULT '{}',
            error_message TEXT,
            error_traceback TEXT,
            build_id INTEGER,
            noise_model TEXT,
            optimization_level INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


@dataclass
class CostRecord(BaseModel):
    """Cost record model."""
    provider: str = ""
    service: str = ""
    resource_type: str = ""
    quantity: float = 0.0
    unit_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    billing_period_start: Optional[datetime] = None
    billing_period_end: Optional[datetime] = None
    build_id: Optional[int] = None
    job_id: Optional[str] = None
    project: str = ""
    environment: str = ""
    cost_center: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "cost_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS cost_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            service TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            quantity REAL DEFAULT 0.0,
            unit_cost REAL DEFAULT 0.0,
            total_cost REAL DEFAULT 0.0,
            currency TEXT DEFAULT 'USD',
            billing_period_start TIMESTAMP,
            billing_period_end TIMESTAMP,
            build_id INTEGER,
            job_id TEXT,
            project TEXT DEFAULT '',
            environment TEXT DEFAULT '',
            cost_center TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


@dataclass
class JobRecord(BaseModel):
    """Job record model."""
    job_id: str = ""
    algorithm_id: str = ""
    provider: str = ""
    backend: str = ""
    shots: int = 0
    priority: str = "medium"
    status: str = "pending"
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = False
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    queue_time_minutes: float = 0.0
    execution_time_minutes: float = 0.0
    build_id: Optional[int] = None
    deployment_id: Optional[str] = None
    circuit_hash: Optional[str] = None
    optimization_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "job_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS job_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            algorithm_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            backend TEXT NOT NULL,
            shots INTEGER DEFAULT 0,
            priority TEXT DEFAULT 'medium',
            status TEXT NOT NULL DEFAULT 'pending',
            scheduled_time TIMESTAMP,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            success BOOLEAN DEFAULT FALSE,
            result_data TEXT DEFAULT '{}',
            error_message TEXT,
            estimated_cost REAL DEFAULT 0.0,
            actual_cost REAL DEFAULT 0.0,
            queue_time_minutes REAL DEFAULT 0.0,
            execution_time_minutes REAL DEFAULT 0.0,
            build_id INTEGER,
            deployment_id TEXT,
            circuit_hash TEXT,
            optimization_level INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


# Utility functions for JSON serialization
def serialize_dict(data: Dict[str, Any]) -> str:
    """Serialize dictionary to JSON string."""
    if not data:
        return "{}"
    return json.dumps(data, default=str)


def deserialize_dict(data: str) -> Dict[str, Any]:
    """Deserialize JSON string to dictionary."""
    if not data or data == "{}":
        return {}
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {}


@dataclass
class DeploymentRecord(BaseModel):
    """Deployment record model."""
    deployment_id: str = ""
    algorithm_id: str = ""
    strategy: str = "blue_green"
    environment: str = "production"
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    validation_fidelity: Optional[float] = None
    validation_error_rate: Optional[float] = None
    validation_cost: float = 0.0
    rollback_reason: Optional[str] = None
    rollback_time: Optional[datetime] = None
    build_id: Optional[int] = None
    circuit_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "deployment_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS deployment_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deployment_id TEXT NOT NULL UNIQUE,
            algorithm_id TEXT NOT NULL,
            strategy TEXT DEFAULT 'blue_green',
            environment TEXT DEFAULT 'production',
            status TEXT NOT NULL DEFAULT 'pending',
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds REAL,
            validation_fidelity REAL,
            validation_error_rate REAL,
            validation_cost REAL DEFAULT 0.0,
            rollback_reason TEXT,
            rollback_time TIMESTAMP,
            build_id INTEGER,
            circuit_hash TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


@dataclass
class ABTestRecord(BaseModel):
    """A/B test record model."""
    test_id: str = ""
    test_name: str = ""
    variant_a: str = ""
    variant_b: str = ""
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    sample_size_a: int = 0
    sample_size_b: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[str] = None
    improvement_percentage: float = 0.0
    p_value: float = 1.0
    confidence_level: float = 0.95
    statistical_significance: bool = False
    recommendation: str = ""
    build_id: Optional[int] = None
    deployment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "ab_test_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS ab_test_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT NOT NULL UNIQUE,
            test_name TEXT NOT NULL,
            variant_a TEXT NOT NULL,
            variant_b TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_hours REAL DEFAULT 0.0,
            sample_size_a INTEGER DEFAULT 0,
            sample_size_b INTEGER DEFAULT 0,
            metrics TEXT DEFAULT '{}',
            winner TEXT,
            improvement_percentage REAL DEFAULT 0.0,
            p_value REAL DEFAULT 1.0,
            confidence_level REAL DEFAULT 0.95,
            statistical_significance BOOLEAN DEFAULT FALSE,
            recommendation TEXT DEFAULT '',
            build_id INTEGER,
            deployment_id TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


@dataclass
class SecurityAuditRecord(BaseModel):
    """Security audit record model."""
    action: str = ""
    resource: str = ""
    user_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    success: bool = True
    failure_reason: Optional[str] = None
    sensitive_data_accessed: bool = False
    permission_level: str = "read"
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "security_audit_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return """
        CREATE TABLE IF NOT EXISTS security_audit_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            resource TEXT NOT NULL,
            user_id TEXT,
            user_agent TEXT,
            ip_address TEXT,
            success BOOLEAN DEFAULT TRUE,
            failure_reason TEXT,
            sensitive_data_accessed BOOLEAN DEFAULT FALSE,
            permission_level TEXT DEFAULT 'read',
            session_id TEXT,
            request_id TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """


def create_all_tables() -> List[str]:
    """Get SQL statements to create all tables."""
    model_classes = [
        BuildRecord,
        HardwareUsageRecord,
        TestResult,
        CostRecord,
        JobRecord,
        DeploymentRecord,
        ABTestRecord,
        SecurityAuditRecord
    ]
    
    return [model_class.create_table_sql() for model_class in model_classes]