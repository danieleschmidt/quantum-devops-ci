"""
Repository layer for database operations in quantum DevOps CI/CD.

This module provides repository classes for managing database operations
with proper typing, error handling, and transaction management.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TypeVar, Generic, Type

from .connection import DatabaseConnection
from .models import (
    BaseModel, BuildRecord, HardwareUsageRecord, TestResult,
    CostRecord, JobRecord, serialize_dict, deserialize_dict
)


T = TypeVar('T', bound=BaseModel)


class BaseRepository(ABC, Generic[T]):
    """Base repository class for database operations."""
    
    def __init__(self, connection: DatabaseConnection):
        """Initialize repository with database connection."""
        self.connection = connection
    
    @property
    @abstractmethod
    def model_class(self) -> Type[T]:
        """Get the model class for this repository."""
        pass
    
    def create(self, entity: T) -> T:
        """Create new entity."""
        data = entity.to_dict()
        entity_id = data.pop('id', None)  # Remove ID for insert
        data['created_at'] = datetime.now().isoformat()
        data['updated_at'] = datetime.now().isoformat()
        
        # Handle JSON serialization
        data = self._serialize_complex_fields(data)
        
        # Build SQL
        columns = list(data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)
        
        sql = f"INSERT INTO {self.model_class.table_name()} ({column_names}) VALUES ({placeholders})"
        
        # Execute and get new ID
        try:
            cursor = self.connection.execute_command(sql, tuple(data.values()))
            if hasattr(cursor, 'lastrowid'):
                entity.id = cursor.lastrowid
            return entity
        except Exception as e:
            raise Exception(f"Failed to create entity: {e}")
    
    def get_by_id(self, entity_id: int) -> Optional[T]:
        """Get entity by ID."""
        sql = f"SELECT * FROM {self.model_class.table_name()} WHERE id = ?"
        results = self.connection.execute_query(sql, (entity_id,))
        
        if not results:
            return None
        
        return self._row_to_entity(results[0])
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """Get all entities with optional pagination."""
        sql = f"SELECT * FROM {self.model_class.table_name()} ORDER BY created_at DESC"
        
        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"
        
        results = self.connection.execute_query(sql)
        return [self._row_to_entity(row) for row in results]
    
    def update(self, entity: T) -> T:
        """Update existing entity."""
        data = entity.to_dict()
        entity_id = data.pop('id')
        data['updated_at'] = datetime.now().isoformat()
        
        # Handle JSON serialization
        data = self._serialize_complex_fields(data)
        
        # Build SQL
        set_clauses = [f"{col} = ?" for col in data.keys()]
        sql = f"UPDATE {self.model_class.table_name()} SET {', '.join(set_clauses)} WHERE id = ?"
        
        params = list(data.values()) + [entity_id]
        self.connection.execute_command(sql, tuple(params))
        
        return entity
    
    def delete(self, entity_id: int) -> bool:
        """Delete entity by ID."""
        sql = f"DELETE FROM {self.model_class.table_name()} WHERE id = ?"
        rows_affected = self.connection.execute_command(sql, (entity_id,))
        return rows_affected > 0
    
    def find(self, **filters) -> List[T]:
        """Find entities by filters."""
        if not filters:
            return self.get_all()
        
        where_clauses = []
        params = []
        
        for key, value in filters.items():
            where_clauses.append(f"{key} = ?")
            params.append(value)
        
        sql = f"SELECT * FROM {self.model_class.table_name()} WHERE {' AND '.join(where_clauses)} ORDER BY created_at DESC"
        results = self.connection.execute_query(sql, tuple(params))
        
        return [self._row_to_entity(row) for row in results]
    
    def count(self, **filters) -> int:
        """Count entities matching filters."""
        if not filters:
            sql = f"SELECT COUNT(*) as count FROM {self.model_class.table_name()}"
            params = ()
        else:
            where_clauses = []
            params = []
            
            for key, value in filters.items():
                where_clauses.append(f"{key} = ?")
                params.append(value)
            
            sql = f"SELECT COUNT(*) as count FROM {self.model_class.table_name()} WHERE {' AND '.join(where_clauses)}"
        
        results = self.connection.execute_query(sql, tuple(params))
        return results[0]['count'] if results else 0
    
    def _serialize_complex_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize complex fields to JSON strings."""
        # Fields that should be serialized as JSON
        json_fields = ['metadata', 'measurement_counts', 'result_data']
        
        for field in json_fields:
            if field in data and isinstance(data[field], (dict, list)):
                data[field] = serialize_dict(data[field])
        
        return data
    
    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """Convert database row to entity object."""
        # Deserialize JSON fields
        json_fields = ['metadata', 'measurement_counts', 'result_data']
        
        for field in json_fields:
            if field in row and isinstance(row[field], str):
                row[field] = deserialize_dict(row[field])
        
        # Convert datetime strings back to datetime objects
        datetime_fields = ['created_at', 'updated_at', 'start_time', 'end_time', 
                          'scheduled_time', 'billing_period_start', 'billing_period_end']
        
        for field in datetime_fields:
            if field in row and row[field] and isinstance(row[field], str):
                try:
                    row[field] = datetime.fromisoformat(row[field])
                except ValueError:
                    # Handle different datetime formats
                    try:
                        row[field] = datetime.strptime(row[field], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        row[field] = None
        
        # Create entity instance
        return self.model_class(**row)


class BuildRepository(BaseRepository[BuildRecord]):
    """Repository for build records."""
    
    @property
    def model_class(self) -> Type[BuildRecord]:
        return BuildRecord
    
    def get_by_commit(self, commit_hash: str) -> List[BuildRecord]:
        """Get builds by commit hash."""
        return self.find(commit_hash=commit_hash)
    
    def get_by_branch(self, branch: str, limit: Optional[int] = None) -> List[BuildRecord]:
        """Get builds by branch."""
        sql = f"SELECT * FROM {self.model_class.table_name()} WHERE branch = ? ORDER BY created_at DESC"
        
        if limit:
            sql += f" LIMIT {limit}"
        
        results = self.connection.execute_query(sql, (branch,))
        return [self._row_to_entity(row) for row in results]
    
    def get_recent_builds(self, days: int = 30) -> List[BuildRecord]:
        """Get builds from recent days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        sql = f"SELECT * FROM {self.model_class.table_name()} WHERE created_at >= ? ORDER BY created_at DESC"
        
        results = self.connection.execute_query(sql, (cutoff_date.isoformat(),))
        return [self._row_to_entity(row) for row in results]
    
    def get_success_rate(self, days: int = 30) -> float:
        """Calculate build success rate for recent period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        total_sql = f"SELECT COUNT(*) as count FROM {self.model_class.table_name()} WHERE created_at >= ?"
        success_sql = f"SELECT COUNT(*) as count FROM {self.model_class.table_name()} WHERE created_at >= ? AND status = 'success'"
        
        total_results = self.connection.execute_query(total_sql, (cutoff_date.isoformat(),))
        success_results = self.connection.execute_query(success_sql, (cutoff_date.isoformat(),))
        
        total_builds = total_results[0]['count'] if total_results else 0
        successful_builds = success_results[0]['count'] if success_results else 0
        
        return successful_builds / total_builds if total_builds > 0 else 0.0


class HardwareUsageRepository(BaseRepository[HardwareUsageRecord]):
    """Repository for hardware usage records."""
    
    @property
    def model_class(self) -> Type[HardwareUsageRecord]:
        return HardwareUsageRecord
    
    def get_by_provider(self, provider: str, days: Optional[int] = None) -> List[HardwareUsageRecord]:
        """Get usage records by provider."""
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            sql = f"SELECT * FROM {self.model_class.table_name()} WHERE provider = ? AND created_at >= ? ORDER BY created_at DESC"
            results = self.connection.execute_query(sql, (provider, cutoff_date.isoformat()))
        else:
            results = self.connection.execute_query(
                f"SELECT * FROM {self.model_class.table_name()} WHERE provider = ? ORDER BY created_at DESC",
                (provider,)
            )
        
        return [self._row_to_entity(row) for row in results]
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get cost summary for recent period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Total cost
        total_sql = f"SELECT SUM(cost_usd) as total_cost, COUNT(*) as total_jobs FROM {self.model_class.table_name()} WHERE created_at >= ?"
        total_results = self.connection.execute_query(total_sql, (cutoff_date.isoformat(),))
        
        # Cost by provider
        provider_sql = f"SELECT provider, SUM(cost_usd) as cost, COUNT(*) as jobs FROM {self.model_class.table_name()} WHERE created_at >= ? GROUP BY provider"
        provider_results = self.connection.execute_query(provider_sql, (cutoff_date.isoformat(),))
        
        # Cost by backend
        backend_sql = f"SELECT backend, SUM(cost_usd) as cost, COUNT(*) as jobs FROM {self.model_class.table_name()} WHERE created_at >= ? GROUP BY backend"
        backend_results = self.connection.execute_query(backend_sql, (cutoff_date.isoformat(),))
        
        total = total_results[0] if total_results else {'total_cost': 0, 'total_jobs': 0}
        
        return {
            'total_cost': total['total_cost'] or 0,
            'total_jobs': total['total_jobs'] or 0,
            'daily_average': (total['total_cost'] or 0) / days,
            'cost_by_provider': {row['provider']: row['cost'] for row in provider_results},
            'cost_by_backend': {row['backend']: row['cost'] for row in backend_results}
        }


class TestResultRepository(BaseRepository[TestResult]):
    """Repository for test results."""
    
    @property
    def model_class(self) -> Type[TestResult]:
        return TestResult
    
    def get_by_build_id(self, build_id: int) -> List[TestResult]:
        """Get test results for a build."""
        return self.find(build_id=build_id)
    
    def get_test_trends(self, test_name: str, days: int = 30) -> List[TestResult]:
        """Get trends for a specific test."""
        cutoff_date = datetime.now() - timedelta(days=days)
        sql = f"SELECT * FROM {self.model_class.table_name()} WHERE test_name = ? AND created_at >= ? ORDER BY created_at ASC"
        
        results = self.connection.execute_query(sql, (test_name, cutoff_date.isoformat()))
        return [self._row_to_entity(row) for row in results]
    
    def get_failure_rate(self, days: int = 30) -> float:
        """Calculate test failure rate."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        total_sql = f"SELECT COUNT(*) as count FROM {self.model_class.table_name()} WHERE created_at >= ?"
        failed_sql = f"SELECT COUNT(*) as count FROM {self.model_class.table_name()} WHERE created_at >= ? AND status = 'failed'"
        
        total_results = self.connection.execute_query(total_sql, (cutoff_date.isoformat(),))
        failed_results = self.connection.execute_query(failed_sql, (cutoff_date.isoformat(),))
        
        total_tests = total_results[0]['count'] if total_results else 0
        failed_tests = failed_results[0]['count'] if failed_results else 0
        
        return failed_tests / total_tests if total_tests > 0 else 0.0


class CostRepository(BaseRepository[CostRecord]):
    """Repository for cost records."""
    
    @property
    def model_class(self) -> Type[CostRecord]:
        return CostRecord
    
    def get_by_project(self, project: str, days: Optional[int] = None) -> List[CostRecord]:
        """Get cost records by project."""
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            sql = f"SELECT * FROM {self.model_class.table_name()} WHERE project = ? AND created_at >= ? ORDER BY created_at DESC"
            results = self.connection.execute_query(sql, (project, cutoff_date.isoformat()))
        else:
            results = self.connection.execute_query(
                f"SELECT * FROM {self.model_class.table_name()} WHERE project = ? ORDER BY created_at DESC",
                (project,)
            )
        
        return [self._row_to_entity(row) for row in results]


class JobRepository(BaseRepository[JobRecord]):
    """Repository for job records."""
    
    @property
    def model_class(self) -> Type[JobRecord]:
        return JobRecord
    
    def get_by_job_id(self, job_id: str) -> Optional[JobRecord]:
        """Get job by job ID."""
        results = self.find(job_id=job_id)
        return results[0] if results else None
    
    def get_by_status(self, status: str) -> List[JobRecord]:
        """Get jobs by status."""
        return self.find(status=status)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status summary."""
        status_sql = f"SELECT status, COUNT(*) as count FROM {self.model_class.table_name()} GROUP BY status"
        status_results = self.connection.execute_query(status_sql)
        
        return {
            'status_counts': {row['status']: row['count'] for row in status_results}
        }
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get job performance metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        metrics_sql = f"""
        SELECT 
            AVG(queue_time_minutes) as avg_queue_time,
            AVG(execution_time_minutes) as avg_execution_time,
            AVG(actual_cost) as avg_cost,
            COUNT(*) as total_jobs,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_jobs
        FROM {self.model_class.table_name()} 
        WHERE created_at >= ?
        """
        
        results = self.connection.execute_query(metrics_sql, (cutoff_date.isoformat(),))
        metrics = results[0] if results else {}
        
        total_jobs = metrics.get('total_jobs', 0)
        successful_jobs = metrics.get('successful_jobs', 0)
        
        return {
            'avg_queue_time_minutes': metrics.get('avg_queue_time', 0) or 0,
            'avg_execution_time_minutes': metrics.get('avg_execution_time', 0) or 0,
            'avg_cost': metrics.get('avg_cost', 0) or 0,
            'total_jobs': total_jobs,
            'success_rate': successful_jobs / total_jobs if total_jobs > 0 else 0.0
        }