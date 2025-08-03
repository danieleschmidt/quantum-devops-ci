"""
Pytest configuration and shared fixtures for quantum DevOps CI/CD tests.

This module provides common test fixtures, configuration, and utilities
used across the test suite.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import Generator, Dict, Any

import pytest

# Add src to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import quantum devops modules
from quantum_devops_ci import check_framework_availability
from quantum_devops_ci.database.connection import DatabaseConnection, DatabaseConfig
from quantum_devops_ci.database.migrations import run_migrations
from quantum_devops_ci.database.cache import CacheManager, MemoryCache
from quantum_devops_ci.testing import NoiseAwareTest
from quantum_devops_ci.monitoring import QuantumCIMonitor
from quantum_devops_ci.cost import CostOptimizer
from quantum_devops_ci.scheduling import QuantumJobScheduler
from quantum_devops_ci.linting import QiskitLinter, LintingConfig

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def framework_availability():
    """Fixture providing framework availability information."""
    return check_framework_availability()


@pytest.fixture(scope="session") 
def test_database():
    """Session-scoped test database."""
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    # Configure test database
    config = DatabaseConfig(
        database_type="sqlite",
        database_name=db_path
    )
    
    # Create connection and run migrations
    connection = DatabaseConnection(config)
    success = run_migrations(connection)
    
    if not success:
        raise RuntimeError("Failed to initialize test database")
    
    yield connection
    
    # Cleanup
    connection.close()
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture(scope="function")
def clean_database(test_database):
    """Function-scoped clean database for each test."""
    # Clean all tables before each test
    tables = [
        'build_records',
        'hardware_usage_records', 
        'test_results',
        'cost_records',
        'job_records'
    ]
    
    for table in tables:
        try:
            test_database.execute_command(f"DELETE FROM {table}")
        except Exception:
            # Table might not exist yet
            pass
    
    yield test_database


@pytest.fixture(scope="function")
def memory_cache():
    """Function-scoped memory cache for testing."""
    cache = MemoryCache(max_size=100)
    yield cache
    cache.clear()


@pytest.fixture(scope="function")
def cache_manager(memory_cache):
    """Function-scoped cache manager for testing."""
    return CacheManager(memory_cache)


@pytest.fixture(scope="function")
def noise_aware_tester():
    """Function-scoped noise-aware test instance."""
    return NoiseAwareTest(default_shots=100, timeout_seconds=30)


@pytest.fixture(scope="function")
def quantum_monitor(clean_database):
    """Function-scoped quantum CI monitor."""
    return QuantumCIMonitor(
        project="test-project",
        local_storage=False  # Use in-memory for testing
    )


@pytest.fixture(scope="function")
def cost_optimizer():
    """Function-scoped cost optimizer."""
    return CostOptimizer(monthly_budget=1000.0)


@pytest.fixture(scope="function")
def job_scheduler():
    """Function-scoped job scheduler."""
    return QuantumJobScheduler(optimization_goal="minimize_cost")


@pytest.fixture(scope="function")
def quantum_linter():
    """Function-scoped quantum linter."""
    config = LintingConfig(
        max_circuit_depth=50,
        max_two_qubit_gates=25,
        max_qubits=10
    )
    return QiskitLinter(config)


@pytest.fixture(scope="function")
def sample_build_data():
    """Sample build data for testing."""
    return {
        'commit': 'abc123def456',
        'branch': 'feature/quantum-test',
        'circuit_count': 3,
        'total_gates': 45,
        'max_depth': 12,
        'estimated_fidelity': 0.92,
        'noise_tests_passed': 8,
        'noise_tests_total': 10,
        'execution_time_seconds': 23.5,
        'metadata': {
            'test_framework': 'pytest',
            'quantum_framework': 'qiskit'
        }
    }


@pytest.fixture(scope="function") 
def sample_hardware_usage():
    """Sample hardware usage data for testing."""
    return {
        'backend': 'ibmq_qasm_simulator',
        'provider': 'ibmq',
        'shots': 1000,
        'queue_time_minutes': 2.5,
        'execution_time_minutes': 1.2,
        'cost_usd': 0.50,
        'circuit_depth': 8,
        'num_qubits': 2,
        'success': True
    }


@pytest.fixture(scope="function")
def sample_experiments():
    """Sample experiments for cost optimization testing."""
    return [
        {
            'id': 'vqe_test',
            'circuit': 'mock_vqe_circuit',
            'shots': 5000,
            'priority': 'high',
            'backend_preferences': ['ibmq_manhattan', 'aws_sv1']
        },
        {
            'id': 'qaoa_test',
            'circuit': 'mock_qaoa_circuit', 
            'shots': 2000,
            'priority': 'medium',
            'backend_preferences': ['aws_sv1']
        },
        {
            'id': 'simple_test',
            'circuit': 'mock_simple_circuit',
            'shots': 500,
            'priority': 'low',
            'backend_preferences': ['qasm_simulator']
        }
    ]


@pytest.fixture(scope="function")
def mock_quantum_circuit():
    """Mock quantum circuit for testing."""
    class MockQuantumCircuit:
        def __init__(self, num_qubits=2):
            self.num_qubits = num_qubits
            self.data = []
            self._depth = 3
        
        def depth(self):
            return self._depth
        
        def measure_all(self):
            pass
        
        def h(self, qubit):
            self.data.append(MockInstruction('h', [qubit]))
        
        def cx(self, control, target):
            self.data.append(MockInstruction('cx', [control, target]))
    
    class MockInstruction:
        def __init__(self, name, qubits):
            self.operation = MockOperation(name)
            self.qubits = [MockQubit(q) for q in qubits]
    
    class MockOperation:
        def __init__(self, name):
            self.name = name
    
    class MockQubit:
        def __init__(self, index):
            self.index = index
    
    return MockQuantumCircuit()


@pytest.fixture(scope="function")
def bell_circuit_qiskit(framework_availability):
    """Bell circuit fixture for Qiskit if available."""
    if not framework_availability['qiskit']:
        pytest.skip("Qiskit not available")
    
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture(scope="function")
def ghz_circuit_qiskit(framework_availability):
    """GHZ circuit fixture for Qiskit if available."""
    if not framework_availability['qiskit']:
        pytest.skip("Qiskit not available")
    
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


@pytest.fixture(scope="function")
def qiskit_backend(framework_availability):
    """Qiskit backend fixture if available."""
    if not framework_availability['qiskit']:
        pytest.skip("Qiskit not available")
    
    try:
        from qiskit.providers.aer import AerSimulator
        return AerSimulator()
    except ImportError:
        pytest.skip("Qiskit Aer not available")


@pytest.fixture(scope="function")
def test_results_data():
    """Sample test results data."""
    return [
        {
            'test_name': 'test_bell_state',
            'framework': 'qiskit',
            'backend': 'qasm_simulator',
            'shots': 1000,
            'fidelity': 0.95,
            'error_rate': 0.02,
            'status': 'passed',
            'execution_time': 1.5
        },
        {
            'test_name': 'test_ghz_state',
            'framework': 'qiskit', 
            'backend': 'qasm_simulator',
            'shots': 1000,
            'fidelity': 0.88,
            'error_rate': 0.05,
            'status': 'passed',
            'execution_time': 2.1
        },
        {
            'test_name': 'test_random_circuit',
            'framework': 'qiskit',
            'backend': 'qasm_simulator',
            'shots': 1000,
            'fidelity': 0.75,
            'error_rate': 0.12,
            'status': 'failed',
            'execution_time': 0.8
        }
    ]


# Test markers
pytest_plugins = ["quantum_devops_ci.pytest_plugin"]


def pytest_configure(config):
    """Configure pytest with quantum-specific settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "mock: mark test as using mocks only"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    # Skip hardware tests by default unless explicitly requested
    if not config.getoption("--hardware"):
        skip_hardware = pytest.mark.skip(reason="Hardware tests disabled")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)
    
    # Skip slow tests unless explicitly requested  
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Slow tests disabled")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add command line options for quantum testing."""
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run tests that require quantum hardware"
    )
    parser.addoption(
        "--slow", 
        action="store_true",
        default=False,
        help="Run slow tests that may take several minutes"
    )
    parser.addoption(
        "--framework",
        choices=["qiskit", "cirq", "pennylane", "all"],
        default="all",
        help="Run tests for specific quantum framework"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment and cleanup after tests."""
    # Setup
    print("\n🧪 Setting up quantum DevOps test environment...")
    
    # Set environment variables for testing
    os.environ['QUANTUM_ENV'] = 'testing'
    os.environ['QUANTUM_LOG_LEVEL'] = 'ERROR'  # Reduce log noise
    
    yield
    
    # Cleanup
    print("\n🧹 Cleaning up test environment...")
    
    # Clean up any test files or resources
    test_cache_dir = Path.home() / '.quantum_devops_ci' / 'test_cache'
    if test_cache_dir.exists():
        import shutil
        shutil.rmtree(test_cache_dir, ignore_errors=True)