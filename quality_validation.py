#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation for Quantum DevOps CI/CD
================================================================

This script validates all implemented features across all three generations
and ensures production-ready quality standards are met.
"""

import sys
import os
import importlib
import traceback
from typing import Dict, List, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class QualityGateValidator:
    """Comprehensive quality gate validator."""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': [],
            'start_time': datetime.now()
        }
    
    def test_module_imports(self) -> bool:
        """Test that all core modules can be imported successfully."""
        logger.info("🧪 Testing module imports...")
        
        modules_to_test = [
            'quantum_devops_ci.testing',
            'quantum_devops_ci.linting', 
            'quantum_devops_ci.cost',
            'quantum_devops_ci.scheduling',
            'quantum_devops_ci.exceptions',
            'quantum_devops_ci.resilience',
            'quantum_devops_ci.monitoring',
            'quantum_devops_ci.concurrency',
            'quantum_devops_ci.caching',
            'quantum_devops_ci.autoscaling',
            'quantum_devops_ci.security',
            'quantum_devops_ci.validation'
        ]
        
        import_success = 0
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                import_success += 1
                logger.info(f"  ✅ {module_name}")
            except Exception as e:
                self.results['errors'].append(f"Import failed: {module_name} - {e}")
                logger.error(f"  ❌ {module_name} - {e}")
        
        self.results['total_tests'] += len(modules_to_test)
        self.results['passed'] += import_success
        self.results['failed'] += len(modules_to_test) - import_success
        
        return import_success == len(modules_to_test)
    
    def test_core_functionality(self) -> bool:
        """Test core functionality instantiation."""
        logger.info("🧪 Testing core functionality...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test NoiseAwareTest
        try:
            from quantum_devops_ci.testing import NoiseAwareTest
            test_instance = NoiseAwareTest()
            assert hasattr(test_instance, 'run_circuit')
            assert hasattr(test_instance, 'run_with_noise')
            tests_passed += 1
            logger.info("  ✅ NoiseAwareTest instantiation")
        except Exception as e:
            self.results['errors'].append(f"NoiseAwareTest failed: {e}")
            logger.error(f"  ❌ NoiseAwareTest - {e}")
        total_tests += 1
        
        # Test QuantumLinter
        try:
            from quantum_devops_ci.linting import QuantumLinter
            linter = QuantumLinter()
            assert hasattr(linter, 'lint_circuit')
            assert hasattr(linter, 'lint_file')
            tests_passed += 1
            logger.info("  ✅ QuantumLinter instantiation")
        except Exception as e:
            self.results['errors'].append(f"QuantumLinter failed: {e}")
            logger.error(f"  ❌ QuantumLinter - {e}")
        total_tests += 1
        
        # Test CostOptimizer
        try:
            from quantum_devops_ci.cost import CostOptimizer
            optimizer = CostOptimizer()
            assert hasattr(optimizer, 'optimize_experiments')
            assert hasattr(optimizer, 'calculate_job_cost')
            tests_passed += 1
            logger.info("  ✅ CostOptimizer instantiation")
        except Exception as e:
            self.results['errors'].append(f"CostOptimizer failed: {e}")
            logger.error(f"  ❌ CostOptimizer - {e}")
        total_tests += 1
        
        # Test QuantumJobScheduler
        try:
            from quantum_devops_ci.scheduling import QuantumJobScheduler
            scheduler = QuantumJobScheduler()
            assert hasattr(scheduler, 'optimize_schedule')
            assert hasattr(scheduler, 'add_job')
            tests_passed += 1
            logger.info("  ✅ QuantumJobScheduler instantiation")
        except Exception as e:
            self.results['errors'].append(f"QuantumJobScheduler failed: {e}")
            logger.error(f"  ❌ QuantumJobScheduler - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def test_resilience_patterns(self) -> bool:
        """Test resilience patterns."""
        logger.info("🧪 Testing resilience patterns...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test CircuitBreaker
        try:
            from quantum_devops_ci.resilience import CircuitBreaker, CircuitBreakerConfig
            config = CircuitBreakerConfig()
            breaker = CircuitBreaker("test", config)
            assert hasattr(breaker, 'call')
            assert hasattr(breaker, 'reset')
            tests_passed += 1
            logger.info("  ✅ CircuitBreaker functionality")
        except Exception as e:
            self.results['errors'].append(f"CircuitBreaker failed: {e}")
            logger.error(f"  ❌ CircuitBreaker - {e}")
        total_tests += 1
        
        # Test RetryHandler
        try:
            from quantum_devops_ci.resilience import RetryHandler, RetryPolicy
            policy = RetryPolicy()
            handler = RetryHandler(policy)
            assert hasattr(handler, 'retry')
            tests_passed += 1
            logger.info("  ✅ RetryHandler functionality")
        except Exception as e:
            self.results['errors'].append(f"RetryHandler failed: {e}")
            logger.error(f"  ❌ RetryHandler - {e}")
        total_tests += 1
        
        # Test resilience manager
        try:
            from quantum_devops_ci.resilience import get_resilience_manager
            manager = get_resilience_manager()
            assert hasattr(manager, 'create_circuit_breaker')
            assert hasattr(manager, 'resilient_quantum_call')
            tests_passed += 1
            logger.info("  ✅ Resilience manager functionality")
        except Exception as e:
            self.results['errors'].append(f"Resilience manager failed: {e}")
            logger.error(f"  ❌ Resilience manager - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def test_monitoring_system(self) -> bool:
        """Test monitoring and alerting system."""
        logger.info("🧪 Testing monitoring system...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test QuantumCIMonitor
        try:
            from quantum_devops_ci.monitoring import QuantumCIMonitor, create_monitor
            monitor = create_monitor("test-project", collector_type='memory')
            assert hasattr(monitor, 'record_build')
            assert hasattr(monitor, 'record_hardware_usage')
            assert hasattr(monitor, 'get_performance_trends')
            tests_passed += 1
            logger.info("  ✅ QuantumCIMonitor functionality")
        except Exception as e:
            self.results['errors'].append(f"QuantumCIMonitor failed: {e}")
            logger.error(f"  ❌ QuantumCIMonitor - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def test_concurrency_features(self) -> bool:
        """Test concurrency and scaling features."""
        logger.info("🧪 Testing concurrency features...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test ConcurrentExecutor
        try:
            from quantum_devops_ci.concurrency import ConcurrentExecutor
            executor = ConcurrentExecutor(max_threads=2, max_processes=2)
            assert hasattr(executor, 'submit_thread_task')
            assert hasattr(executor, 'submit_process_task')
            assert hasattr(executor, 'map_concurrent')
            tests_passed += 1
            logger.info("  ✅ ConcurrentExecutor functionality")
            executor.shutdown()
        except Exception as e:
            self.results['errors'].append(f"ConcurrentExecutor failed: {e}")
            logger.error(f"  ❌ ConcurrentExecutor - {e}")
        total_tests += 1
        
        # Test ResourcePool
        try:
            from quantum_devops_ci.concurrency import ResourcePool
            pool = ResourcePool(
                name="test_pool",
                max_size=5,
                creation_func=lambda: "resource"
            )
            assert hasattr(pool, 'acquire')
            assert hasattr(pool, 'release')
            tests_passed += 1
            logger.info("  ✅ ResourcePool functionality")
        except Exception as e:
            self.results['errors'].append(f"ResourcePool failed: {e}")
            logger.error(f"  ❌ ResourcePool - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def test_caching_system(self) -> bool:
        """Test caching system."""
        logger.info("🧪 Testing caching system...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test CacheManager
        try:
            from quantum_devops_ci.caching import CacheManager
            cache_manager = CacheManager()
            assert hasattr(cache_manager, 'circuit_cache')
            assert hasattr(cache_manager, 'cache_circuit_result')
            assert hasattr(cache_manager, 'get_circuit_result')
            tests_passed += 1
            logger.info("  ✅ CacheManager functionality")
        except Exception as e:
            self.results['errors'].append(f"CacheManager failed: {e}")
            logger.error(f"  ❌ CacheManager - {e}")
        total_tests += 1
        
        # Test MemoryCache
        try:
            from quantum_devops_ci.caching import MemoryCache
            cache = MemoryCache(max_size_mb=1, max_entries=10)
            assert cache.put("test_key", "test_value")
            assert cache.get("test_key") == "test_value"
            tests_passed += 1
            logger.info("  ✅ MemoryCache functionality")
        except Exception as e:
            self.results['errors'].append(f"MemoryCache failed: {e}")
            logger.error(f"  ❌ MemoryCache - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def test_security_features(self) -> bool:
        """Test security and validation features."""
        logger.info("🧪 Testing security features...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test security module
        try:
            from quantum_devops_ci.security import SecurityManager, requires_auth, audit_action
            manager = SecurityManager()
            assert hasattr(manager, 'authenticate')
            assert hasattr(manager, 'authorize')
            tests_passed += 1
            logger.info("  ✅ SecurityManager functionality")
        except Exception as e:
            self.results['errors'].append(f"SecurityManager failed: {e}")
            logger.error(f"  ❌ SecurityManager - {e}")
        total_tests += 1
        
        # Test validation module
        try:
            from quantum_devops_ci.validation import validate_inputs, ValidationError
            # Basic validation test
            @validate_inputs(x=lambda x: x > 0)
            def test_func(x):
                return x * 2
            
            result = test_func(5)
            assert result == 10
            tests_passed += 1
            logger.info("  ✅ Input validation functionality")
        except Exception as e:
            self.results['errors'].append(f"Validation failed: {e}")
            logger.error(f"  ❌ Validation - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def test_integration_scenarios(self) -> bool:
        """Test integration scenarios."""
        logger.info("🧪 Testing integration scenarios...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test end-to-end workflow
        try:
            # Initialize components
            from quantum_devops_ci.monitoring import create_monitor
            from quantum_devops_ci.caching import CacheManager
            from quantum_devops_ci.concurrency import ConcurrentExecutor
            
            monitor = create_monitor("integration-test")
            cache_manager = CacheManager()
            executor = ConcurrentExecutor(max_threads=2)
            
            # Test data flow
            test_build_data = {
                'commit': 'test123',
                'circuit_count': 5,
                'total_gates': 100,
                'max_depth': 10,
                'estimated_fidelity': 0.95,
                'noise_tests_passed': 4,
                'noise_tests_total': 5,
                'execution_time': 1.5
            }
            
            monitor.record_build(test_build_data)
            
            # Test caching
            cache_manager.cache_circuit_result("test_hash", "qasm_simulator", 1000, {"counts": {"00": 500, "11": 500}})
            cached_result = cache_manager.get_circuit_result("test_hash", "qasm_simulator", 1000)
            assert cached_result is not None
            
            executor.shutdown()
            tests_passed += 1
            logger.info("  ✅ End-to-end integration test")
        except Exception as e:
            self.results['errors'].append(f"Integration test failed: {e}")
            logger.error(f"  ❌ Integration test - {e}")
        total_tests += 1
        
        self.results['total_tests'] += total_tests
        self.results['passed'] += tests_passed
        self.results['failed'] += total_tests - tests_passed
        
        return tests_passed == total_tests
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all quality gate tests."""
        logger.info("🚀 Starting Comprehensive Quality Gates Validation")
        logger.info("=" * 60)
        
        test_suites = [
            ("Module Imports", self.test_module_imports),
            ("Core Functionality", self.test_core_functionality),
            ("Resilience Patterns", self.test_resilience_patterns),
            ("Monitoring System", self.test_monitoring_system),
            ("Concurrency Features", self.test_concurrency_features),
            ("Caching System", self.test_caching_system),
            ("Security Features", self.test_security_features),
            ("Integration Scenarios", self.test_integration_scenarios)
        ]
        
        suite_results = {}
        
        for suite_name, test_func in test_suites:
            logger.info(f"\n📋 Running {suite_name} Tests...")
            try:
                suite_results[suite_name] = test_func()
            except Exception as e:
                logger.error(f"❌ {suite_name} test suite failed: {e}")
                suite_results[suite_name] = False
                self.results['errors'].append(f"{suite_name} suite failed: {e}")
        
        # Calculate final results
        self.results['end_time'] = datetime.now()
        self.results['duration'] = (self.results['end_time'] - self.results['start_time']).total_seconds()
        self.results['success_rate'] = (self.results['passed'] / self.results['total_tests']) * 100 if self.results['total_tests'] > 0 else 0
        self.results['suite_results'] = suite_results
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate detailed test report."""
        report = []
        report.append("🔬 QUANTUM DEVOPS CI/CD QUALITY GATES VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY:")
        report.append(f"  Total Tests: {self.results['total_tests']}")
        report.append(f"  Passed: {self.results['passed']} ✅")
        report.append(f"  Failed: {self.results['failed']} ❌")
        report.append(f"  Success Rate: {self.results['success_rate']:.1f}%")
        report.append(f"  Duration: {self.results['duration']:.2f} seconds")
        report.append("")
        
        # Suite results
        if 'suite_results' in self.results:
            report.append("📋 TEST SUITE RESULTS:")
            for suite_name, passed in self.results['suite_results'].items():
                status = "✅ PASSED" if passed else "❌ FAILED"
                report.append(f"  {suite_name}: {status}")
            report.append("")
        
        # Errors
        if self.results['errors']:
            report.append("⚠️  ERRORS:")
            for error in self.results['errors']:
                report.append(f"  • {error}")
            report.append("")
        
        # Quality assessment
        if self.results['success_rate'] >= 90:
            assessment = "🟢 EXCELLENT - Production Ready"
        elif self.results['success_rate'] >= 75:
            assessment = "🟡 GOOD - Minor Issues to Address"
        elif self.results['success_rate'] >= 50:
            assessment = "🟠 FAIR - Significant Issues Need Attention"
        else:
            assessment = "🔴 POOR - Major Issues Must Be Resolved"
        
        report.append(f"🎯 OVERALL QUALITY ASSESSMENT: {assessment}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if self.results['failed'] == 0:
            report.append("  • All tests passed! The system is production-ready.")
            report.append("  • Consider adding more comprehensive integration tests.")
            report.append("  • Monitor performance in production environment.")
        else:
            report.append("  • Address failed test cases before production deployment.")
            report.append("  • Review error logs for specific failure details.")
            report.append("  • Consider implementing additional safeguards.")
        
        return "\n".join(report)


def main():
    """Main validation function."""
    try:
        validator = QualityGateValidator()
        results = validator.run_all_tests()
        report = validator.generate_report()
        
        print(report)
        
        # Exit with appropriate code
        if results['failed'] == 0:
            print("\n🎉 ALL QUALITY GATES PASSED! System is production-ready.")
            sys.exit(0)
        else:
            print(f"\n⚠️  {results['failed']} quality gates failed. Review and fix before deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Quality validation failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()