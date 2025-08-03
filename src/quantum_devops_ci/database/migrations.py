"""
Database migration management for quantum DevOps CI/CD.

This module handles database schema migrations and versioning.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .connection import DatabaseConnection, get_connection
from .models import create_all_tables


class Migration:
    """Represents a database migration."""
    
    def __init__(self, 
                 name: str,
                 version: str,
                 sql: str,
                 description: str = ""):
        self.name = name
        self.version = version
        self.sql = sql
        self.description = description
        self.checksum = hashlib.md5(sql.encode()).hexdigest()
    
    def __str__(self):
        return f"Migration {self.version}: {self.name}"


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, connection: Optional[DatabaseConnection] = None):
        """Initialize migration manager."""
        self.connection = connection or get_connection()
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Ensure migration tracking table exists."""
        sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time_ms INTEGER DEFAULT 0
        )
        """
        self.connection.execute_script(sql)
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations."""
        sql = "SELECT * FROM schema_migrations ORDER BY version"
        return self.connection.execute_query(sql)
    
    def is_migration_applied(self, version: str) -> bool:
        """Check if migration is already applied."""
        sql = "SELECT COUNT(*) as count FROM schema_migrations WHERE version = ?"
        results = self.connection.execute_query(sql, (version,))
        return results[0]['count'] > 0 if results else False
    
    def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        if self.is_migration_applied(migration.version):
            print(f"Migration {migration.version} already applied, skipping")
            return True
        
        print(f"Applying migration {migration.version}: {migration.name}")
        
        start_time = datetime.now()
        
        try:
            # Execute migration SQL
            self.connection.execute_script(migration.sql)
            
            # Record migration as applied
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            record_sql = """
            INSERT INTO schema_migrations (version, name, checksum, execution_time_ms)
            VALUES (?, ?, ?, ?)
            """
            self.connection.execute_command(
                record_sql,
                (migration.version, migration.name, migration.checksum, execution_time)
            )
            
            print(f"Migration {migration.version} applied successfully ({execution_time}ms)")
            return True
            
        except Exception as e:
            print(f"Failed to apply migration {migration.version}: {e}")
            return False
    
    def apply_migrations(self, migrations: List[Migration]) -> bool:
        """Apply multiple migrations in order."""
        success = True
        
        # Sort migrations by version
        sorted_migrations = sorted(migrations, key=lambda m: m.version)
        
        for migration in sorted_migrations:
            if not self.apply_migration(migration):
                success = False
                break
        
        return success
    
    def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration (if rollback SQL provided)."""
        # This is a simplified implementation
        # In practice, you'd need rollback SQL for each migration
        print(f"Rolling back migration {version}")
        
        try:
            # Remove from migrations table
            sql = "DELETE FROM schema_migrations WHERE version = ?"
            rows_affected = self.connection.execute_command(sql, (version,))
            
            if rows_affected > 0:
                print(f"Migration {version} rollback recorded")
                return True
            else:
                print(f"Migration {version} was not applied")
                return False
                
        except Exception as e:
            print(f"Failed to rollback migration {version}: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status summary."""
        applied_migrations = self.get_applied_migrations()
        
        return {
            'applied_count': len(applied_migrations),
            'latest_version': applied_migrations[-1]['version'] if applied_migrations else None,
            'applied_migrations': [
                {
                    'version': m['version'],
                    'name': m['name'],
                    'applied_at': m['applied_at'],
                    'execution_time_ms': m['execution_time_ms']
                }
                for m in applied_migrations
            ]
        }


def create_default_migrations() -> List[Migration]:
    """Create default migrations for the quantum DevOps CI/CD system."""
    migrations = []
    
    # Migration 001: Initial schema
    initial_migration = Migration(
        name="initial_schema",
        version="001",
        description="Create initial database schema for quantum DevOps CI/CD",
        sql="""
        -- Build records table
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
        );
        
        -- Hardware usage records table
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
        );
        
        -- Test results table
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
        );
        
        -- Cost records table
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
        );
        
        -- Job records table
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
        );
        """
    )
    migrations.append(initial_migration)
    
    # Migration 002: Add indexes for performance
    index_migration = Migration(
        name="add_performance_indexes",
        version="002",
        description="Add database indexes for improved query performance",
        sql="""
        -- Indexes for build_records
        CREATE INDEX IF NOT EXISTS idx_build_records_commit_hash ON build_records(commit_hash);
        CREATE INDEX IF NOT EXISTS idx_build_records_branch ON build_records(branch);
        CREATE INDEX IF NOT EXISTS idx_build_records_status ON build_records(status);
        CREATE INDEX IF NOT EXISTS idx_build_records_created_at ON build_records(created_at);
        
        -- Indexes for hardware_usage_records
        CREATE INDEX IF NOT EXISTS idx_hardware_usage_job_id ON hardware_usage_records(job_id);
        CREATE INDEX IF NOT EXISTS idx_hardware_usage_provider ON hardware_usage_records(provider);
        CREATE INDEX IF NOT EXISTS idx_hardware_usage_backend ON hardware_usage_records(backend);
        CREATE INDEX IF NOT EXISTS idx_hardware_usage_build_id ON hardware_usage_records(build_id);
        CREATE INDEX IF NOT EXISTS idx_hardware_usage_created_at ON hardware_usage_records(created_at);
        
        -- Indexes for test_results
        CREATE INDEX IF NOT EXISTS idx_test_results_test_name ON test_results(test_name);
        CREATE INDEX IF NOT EXISTS idx_test_results_status ON test_results(status);
        CREATE INDEX IF NOT EXISTS idx_test_results_build_id ON test_results(build_id);
        CREATE INDEX IF NOT EXISTS idx_test_results_framework ON test_results(framework);
        CREATE INDEX IF NOT EXISTS idx_test_results_created_at ON test_results(created_at);
        
        -- Indexes for cost_records
        CREATE INDEX IF NOT EXISTS idx_cost_records_provider ON cost_records(provider);
        CREATE INDEX IF NOT EXISTS idx_cost_records_project ON cost_records(project);
        CREATE INDEX IF NOT EXISTS idx_cost_records_environment ON cost_records(environment);
        CREATE INDEX IF NOT EXISTS idx_cost_records_created_at ON cost_records(created_at);
        
        -- Indexes for job_records
        CREATE INDEX IF NOT EXISTS idx_job_records_job_id ON job_records(job_id);
        CREATE INDEX IF NOT EXISTS idx_job_records_status ON job_records(status);
        CREATE INDEX IF NOT EXISTS idx_job_records_provider ON job_records(provider);
        CREATE INDEX IF NOT EXISTS idx_job_records_build_id ON job_records(build_id);
        CREATE INDEX IF NOT EXISTS idx_job_records_created_at ON job_records(created_at);
        """
    )
    migrations.append(index_migration)
    
    # Migration 003: Add foreign key constraints (for databases that support them)
    fk_migration = Migration(
        name="add_foreign_keys",
        version="003",
        description="Add foreign key relationships between tables",
        sql="""
        -- Note: SQLite has limited ALTER TABLE support for foreign keys
        -- These would be applied during table creation in practice
        
        -- For now, we'll add some additional constraints and triggers
        CREATE TRIGGER IF NOT EXISTS validate_hardware_usage_build_id
        BEFORE INSERT ON hardware_usage_records
        WHEN NEW.build_id IS NOT NULL
        BEGIN
            SELECT CASE
                WHEN NOT EXISTS (SELECT 1 FROM build_records WHERE id = NEW.build_id)
                THEN RAISE(ABORT, 'Invalid build_id: build record does not exist')
            END;
        END;
        
        CREATE TRIGGER IF NOT EXISTS validate_test_results_build_id
        BEFORE INSERT ON test_results
        WHEN NEW.build_id IS NOT NULL
        BEGIN
            SELECT CASE
                WHEN NOT EXISTS (SELECT 1 FROM build_records WHERE id = NEW.build_id)
                THEN RAISE(ABORT, 'Invalid build_id: build record does not exist')
            END;
        END;
        """
    )
    migrations.append(fk_migration)
    
    return migrations


def run_migrations(connection: Optional[DatabaseConnection] = None) -> bool:
    """Run all default migrations."""
    manager = MigrationManager(connection)
    migrations = create_default_migrations()
    
    print("Running database migrations...")
    success = manager.apply_migrations(migrations)
    
    if success:
        status = manager.get_migration_status()
        print(f"✅ All migrations applied successfully!")
        print(f"Applied {status['applied_count']} migrations")
        print(f"Latest version: {status['latest_version']}")
    else:
        print("❌ Some migrations failed")
    
    return success