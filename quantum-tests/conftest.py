"""
Pytest configuration for quantum tests.

This file provides common fixtures and configuration for all quantum tests
in the quantum-tests directory.
"""

import pytest
import warnings
from pathlib import Path

# Import quantum testing framework
from quantum_devops_ci.testing import NoiseAwareTest
from quantum_devops_ci import AVAILABLE_FRAMEWORKS

# Configure pytest for quantum testing
pytest_plugins = ["quantum_devops_ci.pytest_plugin"]


@pytest.fixture(scope="session", autouse=True)
def configure_quantum_testing():
    """Auto-use fixture to configure quantum testing environment."""
    # Set warning filters for quantum frameworks
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="cirq.*")
    
    print(f"\n🌌 Quantum DevOps CI/CD Test Session")
    print(f"Available frameworks: {list(AVAILABLE_FRAMEWORKS.keys())}")
    
    yield
    
    print(f"\n✅ Quantum test session completed")


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def quantum_test_runner():
    """Provide a configured quantum test runner."""
    return NoiseAwareTest(
        default_shots=1000,
        timeout_seconds=300
    )


# Skip markers for conditional test execution
def pytest_collection_modifyitems(config, items):
    """Add skip markers based on available frameworks and hardware."""
    # Add markers for missing frameworks
    for item in items:
        # Skip Qiskit tests if not available
        if "qiskit" in item.keywords and not AVAILABLE_FRAMEWORKS.get('qiskit'):
            item.add_marker(pytest.mark.skip(reason="Qiskit not available"))
        
        # Skip Cirq tests if not available  
        if "cirq" in item.keywords and not AVAILABLE_FRAMEWORKS.get('cirq'):
            item.add_marker(pytest.mark.skip(reason="Cirq not available"))
            
        # Skip PennyLane tests if not available
        if "pennylane" in item.keywords and not AVAILABLE_FRAMEWORKS.get('pennylane'):
            item.add_marker(pytest.mark.skip(reason="PennyLane not available"))


# Quantum-specific command line options
def pytest_addoption(parser):
    """Add quantum testing command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow quantum tests that may take several minutes"
    )
    
    parser.addoption(
        "--run-hardware",
        action="store_true", 
        default=False,
        help="Run tests that require real quantum hardware access"
    )
    
    parser.addoption(
        "--quantum-test-shots",
        type=int,
        default=1000,
        help="Default number of shots for quantum tests"
    )


def pytest_configure(config):
    """Configure pytest with quantum-specific settings."""
    # Store quantum test configuration
    config.quantum_shots = config.getoption("--quantum-test-shots")
    config.run_slow = config.getoption("--run-slow")
    config.run_hardware = config.getoption("--run-hardware")
    
    # Register additional markers
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"  
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )


def pytest_runtest_setup(item):
    """Setup individual test runs with quantum-specific logic."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Slow test skipped (use --run-slow to run)")
    
    # Skip hardware tests unless explicitly requested
    if "hardware" in item.keywords and not item.config.getoption("--run-hardware"):
        pytest.skip("Hardware test skipped (use --run-hardware to run)")


# Quantum test result reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add quantum-specific summary to test report."""
    if hasattr(config, '_quantum_stats'):
        stats = config._quantum_stats
        terminalreporter.write_sep("=", "Quantum Test Summary")
        
        for framework, count in stats.get('framework_usage', {}).items():
            terminalreporter.write_line(f"{framework}: {count} tests")
        
        total_shots = stats.get('total_shots', 0)
        if total_shots > 0:
            terminalreporter.write_line(f"Total quantum shots: {total_shots:,}")
        
        execution_time = stats.get('total_execution_time', 0)
        if execution_time > 0:
            terminalreporter.write_line(f"Total execution time: {execution_time:.2f}s")