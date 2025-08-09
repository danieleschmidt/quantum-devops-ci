"""
Database connection management for quantum DevOps CI/CD.

This module provides database connection handling with support for
SQLite (default), PostgreSQL, and other databases.
"""

import os
import sqlite3
import warnings
from typing import Optional, Dict, Any, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
import threading

try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_type: str = "sqlite"  # sqlite, postgresql, mysql
    database_name: str = "quantum_devops.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_pool_size: int = 5
    max_connections: int = 20
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        return cls(
            database_type=os.getenv('QUANTUM_DB_TYPE', 'sqlite'),
            database_name=os.getenv('QUANTUM_DB_NAME', 'quantum_devops.db'),
            host=os.getenv('QUANTUM_DB_HOST'),
            port=int(os.getenv('QUANTUM_DB_PORT', '5432')),
            username=os.getenv('QUANTUM_DB_USER'),
            password=os.getenv('QUANTUM_DB_PASSWORD'),
            connection_pool_size=int(os.getenv('QUANTUM_DB_POOL_SIZE', '5')),
            max_connections=int(os.getenv('QUANTUM_DB_MAX_CONNECTIONS', '20')),
            timeout=int(os.getenv('QUANTUM_DB_TIMEOUT', '30'))
        )
    
    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        if self.database_type == "sqlite":
            return f"sqlite:///{self.database_name}"
        elif self.database_type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
        elif self.database_type == "mysql":
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


class DatabaseConnection:
    """Database connection manager with connection pooling."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection manager.
        
        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig.from_env()
        self._connection_pool = None
        self._local_storage = threading.local()
        self._init_lock = threading.Lock()
        
        # Ensure data directory exists for SQLite
        if self.config.database_type == "sqlite":
            db_path = Path(self.config.database_name)
            if not db_path.is_absolute():
                # Store in user data directory
                data_dir = Path.home() / '.quantum_devops_ci' / 'data'
                data_dir.mkdir(parents=True, exist_ok=True)
                self.config.database_name = str(data_dir / self.config.database_name)
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        if self._connection_pool is not None:
            return
        
        with self._init_lock:
            if self._connection_pool is not None:
                return
            
            if self.config.database_type == "postgresql":
                if not POSTGRES_AVAILABLE:
                    raise ImportError("psycopg2 not available for PostgreSQL connections")
                
                self._connection_pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=self.config.max_connections,
                    database=self.config.database_name,
                    user=self.config.username,
                    password=self.config.password,
                    host=self.config.host,
                    port=self.config.port
                )
            # For SQLite and MySQL, we'll use direct connections
    
    def get_connection(self):
        """Get database connection."""
        if self.config.database_type == "sqlite":
            return self._get_sqlite_connection()
        elif self.config.database_type == "postgresql":
            return self._get_postgresql_connection()
        elif self.config.database_type == "mysql":
            return self._get_mysql_connection()
        else:
            raise ValueError(f"Unsupported database type: {self.config.database_type}")
    
    def _get_sqlite_connection(self):
        """Get SQLite connection."""
        # Use thread-local storage for SQLite connections
        if not hasattr(self._local_storage, 'connection') or self._local_storage.connection is None:
            try:
                connection = sqlite3.connect(
                    self.config.database_name,
                    timeout=self.config.timeout,
                    check_same_thread=False
                )
                connection.row_factory = sqlite3.Row  # Enable dict-like access
                # Enable foreign keys and WAL mode for better performance
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA journal_mode = WAL")
                connection.commit()
                self._local_storage.connection = connection
            except sqlite3.Error as e:
                raise RuntimeError(f"Failed to connect to SQLite database: {e}")
        
        return self._local_storage.connection
    
    def _get_postgresql_connection(self):\n        \"\"\"Get PostgreSQL connection from pool.\"\"\"\n        if not POSTGRES_AVAILABLE:\n            raise ImportError(\"psycopg2 not available for PostgreSQL connections\")\n        \n        self._initialize_pool()\n        return self._connection_pool.getconn()\n    \n    def _get_mysql_connection(self):\n        \"\"\"Get MySQL connection.\"\"\"\n        if not MYSQL_AVAILABLE:\n            raise ImportError(\"mysql-connector not available for MySQL connections\")\n        \n        return mysql.connector.connect(\n            host=self.config.host,\n            port=self.config.port,\n            user=self.config.username,\n            password=self.config.password,\n            database=self.config.database_name,\n            autocommit=True\n        )\n    \n    def return_connection(self, connection):\n        \"\"\"Return connection to pool (for PostgreSQL).\"\"\"\n        if self.config.database_type == \"postgresql\" and self._connection_pool:\n            self._connection_pool.putconn(connection)\n        elif self.config.database_type in [\"sqlite\", \"mysql\"]:\n            # For SQLite (thread-local) and MySQL, we keep connections open\n            pass\n    \n    @contextmanager\n    def get_session(self):\n        \"\"\"Get database session with automatic cleanup.\"\"\"\n        connection = self.get_connection()\n        try:\n            yield connection\n            if hasattr(connection, 'commit'):\n                connection.commit()\n        except Exception as e:\n            if hasattr(connection, 'rollback'):\n                connection.rollback()\n            raise\n        finally:\n            self.return_connection(connection)\n    \n    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:\n        \"\"\"Execute SELECT query and return results.\"\"\"\n        with self.get_session() as conn:\n            cursor = conn.cursor()\n            cursor.execute(query, params or ())\n            \n            if self.config.database_type == \"sqlite\":\n                # SQLite with Row factory returns dict-like objects\n                return [dict(row) for row in cursor.fetchall()]\n            else:\n                # For other databases, get column names and create dicts\n                columns = [desc[0] for desc in cursor.description]\n                return [dict(zip(columns, row)) for row in cursor.fetchall()]\n    \n    def execute_command(self, command: str, params: Optional[tuple] = None) -> int:\n        \"\"\"Execute INSERT/UPDATE/DELETE command and return affected rows.\"\"\"\n        with self.get_session() as conn:\n            cursor = conn.cursor()\n            cursor.execute(command, params or ())\n            return cursor.rowcount\n    \n    def execute_script(self, script: str):\n        \"\"\"Execute SQL script (multiple statements).\"\"\"\n        with self.get_session() as conn:\n            if self.config.database_type == \"sqlite\":\n                conn.executescript(script)\n            else:\n                # For other databases, split and execute statements\n                statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]\n                cursor = conn.cursor()\n                for statement in statements:\n                    cursor.execute(statement)\n    \n    def table_exists(self, table_name: str) -> bool:\n        \"\"\"Check if table exists in database.\"\"\"\n        if self.config.database_type == \"sqlite\":\n            query = \"SELECT name FROM sqlite_master WHERE type='table' AND name=?\"\n        elif self.config.database_type == \"postgresql\":\n            query = \"SELECT tablename FROM pg_tables WHERE tablename=%s\"\n        elif self.config.database_type == \"mysql\":\n            query = \"SELECT table_name FROM information_schema.tables WHERE table_name=%s AND table_schema=DATABASE()\"\n        else:\n            return False\n        \n        results = self.execute_query(query, (table_name,))\n        return len(results) > 0\n    \n    def get_table_schema(self, table_name: str) -> Dict[str, Any]:\n        \"\"\"Get table schema information.\"\"\"\n        if self.config.database_type == \"sqlite\":\n            query = f\"PRAGMA table_info({table_name})\"\n            results = self.execute_query(query)\n            return {\n                'columns': [\n                    {\n                        'name': row['name'],\n                        'type': row['type'],\n                        'nullable': not row['notnull'],\n                        'primary_key': bool(row['pk'])\n                    }\n                    for row in results\n                ]\n            }\n        else:\n            # For other databases, would need specific schema queries\n            warnings.warn(f\"Schema inspection not implemented for {self.config.database_type}\")\n            return {'columns': []}\n    \n    def close(self):\n        \"\"\"Close database connections.\"\"\"\n        if hasattr(self._local_storage, 'connection'):\n            self._local_storage.connection.close()\n            delattr(self._local_storage, 'connection')\n        \n        if self._connection_pool:\n            self._connection_pool.closeall()\n            self._connection_pool = None\n\n\n# Global connection instance\n_global_connection: Optional[DatabaseConnection] = None\n_connection_lock = threading.Lock()\n\n\ndef get_connection(config: Optional[DatabaseConfig] = None) -> DatabaseConnection:\n    \"\"\"Get global database connection instance.\"\"\"\n    global _global_connection\n    \n    if _global_connection is None:\n        with _connection_lock:\n            if _global_connection is None:\n                _global_connection = DatabaseConnection(config)\n    \n    return _global_connection\n\n\ndef close_connection():\n    \"\"\"Close global database connection.\"\"\"\n    global _global_connection\n    \n    if _global_connection is not None:\n        with _connection_lock:\n            if _global_connection is not None:\n                _global_connection.close()\n                _global_connection = None