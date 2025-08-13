"""
Simplified Generation 4 Intelligence Validation Suite.

This test suite focuses on core functionality without decorators
to validate Generation 4 features.
"""

import sys
import os
import warnings
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Test results tracking
test_results = {
    'total_tests': 0,
    'passed_tests': 0,
    'failed_tests': 0,
    'test_details': []
}

def run_test(test_name: str, test_function):
    """Run individual test with error handling."""
    global test_results
    test_results['total_tests'] += 1
    
    try:
        test_function()
        test_results['passed_tests'] += 1
        test_results['test_details'].append({
            'name': test_name,
            'status': 'PASSED',
            'error': None
        })
        print(f"✅ {test_name}")
        return True
    except Exception as e:
        test_results['failed_tests'] += 1
        test_results['test_details'].append({
            'name': test_name,
            'status': 'FAILED', 
            'error': str(e)
        })
        print(f"❌ {test_name}: {e}")
        return False


def test_ml_optimization_core():
    """Test core ML optimization functionality."""
    from quantum_devops_ci.ml_optimization import (
        OptimizationObjective, CircuitMetrics
    )
    
    # Test metric creation
    metrics = CircuitMetrics(
        depth=10,
        gate_count=50,
        two_qubit_gates=15,
        estimated_error_rate=0.01,
        estimated_execution_time=1.0,
        estimated_cost=0.1,
        fidelity_score=0.95
    )
    
    # Test feature vector conversion
    features = metrics.to_feature_vector()
    assert len(features) == 8
    assert features[0] == 10.0  # depth
    assert features[1] == 50.0  # gate_count


def test_circuit_optimization():
    """Test circuit optimization without decorators."""
    from quantum_devops_ci.ml_optimization import (
        QuantumCircuitOptimizer, OptimizationObjective
    )
    
    optimizer = QuantumCircuitOptimizer()
    
    # Mock circuit
    class MockCircuit:
        def depth(self):
            return 20
        def count_ops(self):
            return {'total': 100}
    
    circuit = MockCircuit()
    
    # Test simple optimization
    result = optimizer._simple_optimization(
        circuit, OptimizationObjective.MINIMIZE_DEPTH, 5
    )
    
    optimized_circuit, metrics, history = result
    assert len(history) == 5
    assert metrics.depth > 0


def test_quantum_error_correction_core():
    """Test QEC core functionality."""
    from quantum_devops_ci.quantum_error_correction import (
        QECCode, ErrorSyndrome, LogicalQubit, SurfaceCodeDecoder
    )
    
    # Test logical qubit creation
    logical_qubit = LogicalQubit(
        code_type=QECCode.SURFACE_CODE,
        physical_qubits=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        data_qubits=[0, 2, 6, 8],
        ancilla_qubits=[1, 3, 5, 7],
        distance=3
    )
    
    assert logical_qubit.distance == 3
    assert len(logical_qubit.physical_qubits) == 9
    assert 'X' in logical_qubit.logical_operators
    assert 'Z' in logical_qubit.logical_operators
    
    # Test syndrome creation
    syndrome = ErrorSyndrome(
        syndrome_bits=[1, 0, 1],
        measurement_round=1,
        timestamp=datetime.now()
    )
    
    assert syndrome.hamming_weight() == 2
    assert syndrome.to_binary_string() == '101'


def test_surface_code_decoder():
    """Test surface code decoder."""
    from quantum_devops_ci.quantum_error_correction import (
        SurfaceCodeDecoder, ErrorSyndrome
    )
    
    decoder = SurfaceCodeDecoder(distance=3)
    
    # Test syndrome decoding
    syndrome = ErrorSyndrome(
        syndrome_bits=[0, 0, 1],
        measurement_round=1,
        timestamp=datetime.now()
    )
    
    errors = decoder.decode_syndrome(syndrome)
    assert isinstance(errors, list)


def test_predictive_analytics_core():
    """Test predictive analytics core functionality."""
    from quantum_devops_ci.predictive_analytics import (
        TimeSeriesData, PredictionResult, PredictionHorizon, MetricType
    )
    
    # Test time series data
    data = TimeSeriesData(
        timestamp=datetime.now(),
        value=42.0,
        metadata={'provider': 'test'}
    )
    
    assert data.value == 42.0
    
    # Test prediction result
    result = PredictionResult(
        predicted_value=100.0,
        confidence_intervals=(90.0, 110.0),
        prediction_horizon=PredictionHorizon.HOUR,
        model_confidence=0.85
    )
    
    assert result.is_reliable()
    assert result.predicted_value == 100.0


def test_cost_predictor_core():
    """Test cost predictor core functionality."""
    from quantum_devops_ci.predictive_analytics import QuantumCostPredictor
    
    predictor = QuantumCostPredictor()
    
    # Add some data
    predictor.add_historical_data(
        provider='test_provider',
        backend='test_backend',
        timestamp=datetime.now(),
        cost=10.0,
        features={'shots': 1000, 'circuit_depth': 10, 'gate_count': 50, 'priority_multiplier': 1.0}
    )
    
    # Check data was added
    assert 'test_provider_test_backend' in predictor.training_data
    assert len(predictor.training_data['test_provider_test_backend']) == 1


def test_advanced_validation_core():
    """Test advanced validation core functionality."""
    from quantum_devops_ci.advanced_validation import (
        ValidationMetrics, ValidationLevel, ExperimentalDesign
    )
    
    # Test validation metrics
    metrics = ValidationMetrics(
        accuracy=0.95,
        precision=0.90,
        recall=0.88,
        f1_score=0.89,
        p_value=0.03
    )
    
    assert metrics.is_statistically_significant()
    assert not metrics.meets_publication_standards()  # Missing statistical power
    
    # Test experimental design
    design = ExperimentalDesign(
        hypothesis="Algorithm performs better than baseline",
        null_hypothesis="No difference from baseline", 
        alternative_hypothesis="Algorithm is better"
    )
    
    sample_size = design.calculate_required_sample_size()
    assert sample_size > 0


def test_quantum_algorithm_validator_core():
    """Test quantum algorithm validator core functionality."""
    from quantum_devops_ci.advanced_validation import (
        QuantumAlgorithmValidator, ValidationLevel
    )
    import numpy as np
    
    validator = QuantumAlgorithmValidator(ValidationLevel.STANDARD)
    
    # Test basic metrics calculation
    predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    actual = [1.1, 1.9, 3.1, 3.9, 5.1]
    
    metrics = validator._calculate_basic_metrics(predicted, actual)
    
    assert metrics.accuracy >= 0
    assert metrics.precision >= 0


def test_quantum_sovereignty_core():
    """Test quantum sovereignty core functionality."""
    from quantum_devops_ci.quantum_sovereignty import (
        SovereigntyLevel, TechnologyClassification, SovereigntyPolicy,
        AccessRequest, QuantumSovereigntyManager
    )
    
    # Test policy creation
    policy = SovereigntyPolicy(
        country_code='US',
        sovereignty_level=SovereigntyLevel.CONTROLLED,
        allowed_destinations=['CA', 'GB'],
        restricted_destinations=['CN', 'RU']
    )
    
    assert policy.country_code == 'US'
    assert len(policy.technology_restrictions) > 0
    
    # Test access request
    request = AccessRequest(
        request_id='test_001',
        requester_info={'country': 'US', 'organization': 'TestOrg'},
        requested_technology=TechnologyClassification.DUAL_USE,
        intended_use='Research',
        destination_country='CA',
        duration=timedelta(days=90),
        justification='Academic collaboration'
    )
    
    risk_score = request.risk_score()
    assert 0 <= risk_score <= 1


def test_sovereignty_manager_core():
    """Test sovereignty manager core functionality."""
    from quantum_devops_ci.quantum_sovereignty import (
        QuantumSovereigntyManager, TechnologyClassification
    )
    
    manager = QuantumSovereigntyManager()
    
    # Test technology classification
    classification = manager.classify_quantum_technology(
        algorithm_description="Machine learning optimization algorithm",
        intended_application="Business optimization",
        technical_parameters={'qubits': 10}
    )
    
    assert classification in [
        TechnologyClassification.DUAL_USE,
        TechnologyClassification.COMMERCIAL,
        TechnologyClassification.APPLIED_RESEARCH
    ]
    
    # Test blocked entities
    manager.add_blocked_entity('BadActor Inc.', 'Security violation')
    assert 'BadActor Inc.' in manager.blocked_entities


def test_ml_predictor_statistical_fallback():
    """Test ML predictor statistical fallback."""
    from quantum_devops_ci.ml_optimization import QuantumMLPredictor, CircuitMetrics
    
    predictor = QuantumMLPredictor()
    
    # Add training data
    for i in range(15):
        metrics = CircuitMetrics(
            depth=10 + i,
            gate_count=50 + i * 2,
            two_qubit_gates=5 + i,
            estimated_error_rate=0.01,
            estimated_execution_time=1.0,
            estimated_cost=0.1
        )
        predictor.add_training_data(metrics, 'test_metric', float(i))
    
    # Train with statistical fallback
    scores = predictor._train_statistical_models(['test_metric'])
    assert 'test_metric' in scores


def run_all_tests():
    """Run all simplified Generation 4 validation tests."""
    print("🚀 Starting Generation 4 Simplified Validation Suite")
    print("=" * 60)
    
    # Core functionality tests
    print("\n🔬 Core Functionality Tests:")
    run_test("ML Optimization Core", test_ml_optimization_core)
    run_test("Circuit Optimization", test_circuit_optimization)
    run_test("QEC Core Functionality", test_quantum_error_correction_core)
    run_test("Surface Code Decoder", test_surface_code_decoder)
    run_test("Predictive Analytics Core", test_predictive_analytics_core)
    run_test("Cost Predictor Core", test_cost_predictor_core)
    run_test("Advanced Validation Core", test_advanced_validation_core)
    run_test("Algorithm Validator Core", test_quantum_algorithm_validator_core)
    run_test("Quantum Sovereignty Core", test_quantum_sovereignty_core)
    run_test("Sovereignty Manager Core", test_sovereignty_manager_core)
    run_test("ML Statistical Fallback", test_ml_predictor_statistical_fallback)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 GENERATION 4 SIMPLIFIED VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']} ✅")
    print(f"Failed: {test_results['failed_tests']} ❌")
    
    success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Generation 4 core features are production-ready!")
    elif success_rate >= 80:
        print("🟢 GOOD: Generation 4 core features are mostly functional")  
    elif success_rate >= 70:
        print("🟡 ACCEPTABLE: Generation 4 core features need minor fixes")
    else:
        print("🔴 NEEDS WORK: Generation 4 core features require improvements")
        
    # Show failed tests
    failed_tests = [t for t in test_results['test_details'] if t['status'] == 'FAILED']
    if failed_tests:
        print(f"\n❌ Failed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - {test['name']}: {test['error']}")
    
    return test_results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results['failed_tests'] == 0 else 1
    sys.exit(exit_code)