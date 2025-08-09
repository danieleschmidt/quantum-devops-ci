#!/usr/bin/env python3
"""
Minimal test of the quantum DevOps CI framework without external dependencies.

This demonstrates the database and core infrastructure functionality.
"""

import sys
import os
from pathlib import Path

# Add the source directory to Python path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test basic module imports."""
    print("🔍 Testing module imports...")
    
    try:
        from quantum_devops_ci.database.connection import DatabaseConnection, DatabaseConfig
        print("✅ Database connection module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import database connection: {e}")
        return False
    
    try:
        from quantum_devops_ci.database.migrations import run_migrations, MigrationManager
        print("✅ Database migrations module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import database migrations: {e}")
        return False
    
    try:
        from quantum_devops_ci.testing import NoiseAwareTest, TestResult
        print("✅ Testing framework imported successfully")
    except Exception as e:
        print(f"❌ Failed to import testing framework: {e}")
        return False
        
    return True


def test_database():
    """Test database functionality."""
    print("\n🗄️  Testing database functionality...")
    
    try:
        from quantum_devops_ci.database.connection import DatabaseConnection, DatabaseConfig
        from quantum_devops_ci.database.migrations import run_migrations
        
        # Create in-memory database for testing
        config = DatabaseConfig(database_type="sqlite", database_name=":memory:")
        connection = DatabaseConnection(config)
        
        print("✅ Database connection created successfully")
        
        # Test basic connection
        result = connection.execute_query("SELECT 1 as test")
        if result and result[0]['test'] == 1:
            print("✅ Database query test passed")
        else:
            print("❌ Database query test failed")
            return False
        
        # Run migrations
        success = run_migrations(connection)
        if success:
            print("✅ Database migrations completed successfully")
        else:
            print("❌ Database migrations failed")
            return False
        
        # Test table creation
        tables = connection.execute_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [row['name'] for row in tables]
        
        expected_tables = ['build_records', 'hardware_usage_records', 'test_results', 'cost_records', 'job_records', 'schema_migrations']
        for table in expected_tables:
            if table in table_names:
                print(f"✅ Table '{table}' created successfully")
            else:
                print(f"❌ Table '{table}' not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_cli_basic():
    """Test CLI basic functionality."""
    print("\n💻 Testing CLI functionality...")
    
    try:
        from quantum_devops_ci.cli import create_parser
        
        parser = create_parser()
        
        # Test help
        args = parser.parse_args(['--help'])
        print("✅ CLI parser created successfully")
        return True
        
    except SystemExit:
        # Expected for --help
        print("✅ CLI help system working")
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_testing_framework_basic():
    """Test basic testing framework without quantum dependencies."""
    print("\n🧪 Testing framework (basic functionality)...")
    
    try:
        from quantum_devops_ci.testing import NoiseAwareTest, TestResult
        
        # Create test runner
        test_runner = NoiseAwareTest(default_shots=100)
        print("✅ NoiseAwareTest instance created successfully")
        
        # Test result creation
        result = TestResult(
            counts={'00': 50, '11': 50},
            shots=100,
            execution_time=0.1,
            backend_name="test_backend"
        )
        
        # Test fidelity calculation
        fidelity = test_runner.calculate_bell_fidelity(result)
        if 0.9 <= fidelity <= 1.0:
            print(f"✅ Bell fidelity calculation working: {fidelity:.3f}")
        else:
            print(f"❌ Unexpected fidelity value: {fidelity:.3f}")
            return False
        
        print("✅ Testing framework basic functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Testing framework test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        config_file = Path(__file__).parent.parent.parent / "quantum.config.yml"
        
        if config_file.exists():
            print(f"✅ Configuration file found: {config_file}")
            
            # Try to read with basic YAML parsing if available
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'hardware_access' in config:
                    print("✅ Configuration structure looks valid")
                else:
                    print("⚠️  Configuration structure may be incomplete")
                    
            except ImportError:
                print("⚠️  YAML not available, can't validate config structure")
                
        else:
            print("⚠️  Configuration file not found (this is okay for testing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🚀 Quantum DevOps CI - Basic Functionality Test")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Module Imports", test_imports()))
    test_results.append(("Database", test_database()))
    test_results.append(("CLI Basic", test_cli_basic()))
    test_results.append(("Testing Framework", test_testing_framework_basic()))
    test_results.append(("Configuration", test_configuration()))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! Core functionality is working.")
        print("\n📚 Next Steps:")
        print("1. Install Qiskit: pip install qiskit qiskit-aer")
        print("2. Run: python3 examples/basic/quantum_test_demo.py")
        print("3. Run: quantum-test init")
        print("4. Run: quantum-test run --framework qiskit")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())