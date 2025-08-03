"""
Database and persistence layer for quantum DevOps CI/CD.

This module provides database connectivity, schema management, and
data persistence capabilities for the quantum CI/CD toolkit.
"""

from .connection import DatabaseConnection, get_connection
from .models import (
    BaseModel,
    BuildRecord,
    HardwareUsageRecord,
    TestResult,
    CostRecord,
    JobRecord
)
from .repositories import (
    BaseRepository,
    BuildRepository,
    HardwareUsageRepository,
    TestResultRepository,
    CostRepository,
    JobRepository
)
from .migrations import MigrationManager
from .cache import CacheManager

__all__ = [
    # Connection management
    'DatabaseConnection',
    'get_connection',
    
    # Models
    'BaseModel',
    'BuildRecord',
    'HardwareUsageRecord', 
    'TestResult',
    'CostRecord',
    'JobRecord',
    
    # Repositories
    'BaseRepository',
    'BuildRepository',
    'HardwareUsageRepository',
    'TestResultRepository',
    'CostRepository',
    'JobRepository',
    
    # Utilities
    'MigrationManager',
    'CacheManager'
]