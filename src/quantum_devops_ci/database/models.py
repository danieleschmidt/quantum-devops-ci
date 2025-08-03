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
    """Model for build/CI records."""
    commit_hash: str = ""
    branch: str = ""
    build_number: Optional[int] = None
    status: str = "pending"  # pending, running, success, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Quantum-specific metrics
    circuit_count: int = 0
    total_gates: int = 0
    max_circuit_depth: int = 0
    estimated_fidelity: float = 0.0
    
    # Test results
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    noise_tests_passed: int = 0
    noise_tests_total: int = 0
    
    # Metadata
    framework: str = ""
    backend: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "build_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return \"\"\"
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
        \"\"\"


@dataclass
class HardwareUsageRecord(BaseModel):
    """Model for quantum hardware usage records."""
    job_id: str = ""
    provider: str = ""
    backend: str = ""
    
    # Usage metrics
    shots: int = 0
    circuit_depth: Optional[int] = None
    num_qubits: Optional[int] = None
    
    # Timing
    queue_time_minutes: float = 0.0
    execution_time_minutes: float = 0.0
    
    # Cost
    cost_usd: float = 0.0
    cost_currency: str = "USD"
    
    # Results
    success: bool = True
    error_message: Optional[str] = None
    
    # Metadata
    build_id: Optional[int] = None
    experiment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "hardware_usage_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return \"\"\"
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
        \"\"\"


@dataclass
class TestResult(BaseModel):
    """Model for quantum test results."""
    test_name: str = ""
    test_class: str = ""
    test_file: str = ""
    
    # Test execution
    status: str = "pending"  # pending, running, passed, failed, skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Quantum-specific results
    framework: str = ""
    backend: str = ""
    shots: int = 0
    fidelity: Optional[float] = None
    error_rate: Optional[float] = None
    
    # Results data
    measurement_counts: Dict[str, int] = field(default_factory=dict)
    
    # Error info
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Associations
    build_id: Optional[int] = None
    
    # Metadata
    noise_model: Optional[str] = None
    optimization_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "test_results"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return \"\"\"
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
        \"\"\"


@dataclass
class CostRecord(BaseModel):
    """Model for cost tracking records."""
    provider: str = ""
    service: str = ""  # quantum_computing, storage, data_transfer, etc.
    resource_type: str = ""  # shots, minutes, GB, etc.
    
    # Usage
    quantity: float = 0.0
    unit_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    
    # Time period
    billing_period_start: Optional[datetime] = None
    billing_period_end: Optional[datetime] = None
    
    # Associations
    build_id: Optional[int] = None
    job_id: Optional[str] = None
    
    # Categorization
    project: str = ""
    environment: str = ""  # development, staging, production
    cost_center: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "cost_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return \"\"\"
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
        \"\"\"


@dataclass
class JobRecord(BaseModel):
    """Model for quantum job records."""
    job_id: str = ""
    algorithm_id: str = ""
    
    # Job configuration
    provider: str = ""
    backend: str = ""
    shots: int = 0
    priority: str = "medium"  # low, medium, high, critical
    
    # Scheduling
    status: str = "pending"  # pending, queued, running, completed, failed, cancelled
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    success: bool = False
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Cost and performance
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    queue_time_minutes: float = 0.0
    execution_time_minutes: float = 0.0
    
    # Associations
    build_id: Optional[int] = None
    deployment_id: Optional[str] = None
    
    # Metadata
    circuit_hash: Optional[str] = None
    optimization_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def table_name(cls) -> str:
        return "job_records"
    
    @classmethod
    def create_table_sql(cls) -> str:
        return \"\"\"
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
        \"\"\"


# Helper functions for JSON serialization
def serialize_dict(data: Dict[str, Any]) -> str:
    """Serialize dictionary to JSON string for database storage."""
    return json.dumps(data, default=str)


def deserialize_dict(json_str: str) -> Dict[str, Any]:
    """Deserialize JSON string from database to dictionary."""
    try:
        return json.loads(json_str) if json_str else {}
    except json.JSONDecodeError:
        return {}


# Model registry for dynamic table creation
MODEL_REGISTRY = [
    BuildRecord,
    HardwareUsageRecord,
    TestResult,
    CostRecord,
    JobRecord
]


def get_all_models() -> List[type]:
    """Get all registered model classes."""
    return MODEL_REGISTRY


def create_all_tables(connection) -> None:
    """Create all model tables in the database."""
    for model_class in MODEL_REGISTRY:
        sql = model_class.create_table_sql()
        connection.execute_script(sql)