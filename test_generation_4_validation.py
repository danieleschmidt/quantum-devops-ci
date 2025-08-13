"""
Comprehensive Generation 4 Intelligence Validation Suite.

This test suite validates all Generation 4 features including ML optimization,
quantum error correction, predictive analytics, advanced validation, and
quantum sovereignty controls.
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

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


def test_ml_optimization_imports():
    """Test ML optimization module imports."""
    from quantum_devops_ci.ml_optimization import (
        OptimizationObjective,
        CircuitMetrics,
        OptimizationResult,
        QuantumMLPredictor,
        QuantumCircuitOptimizer,
        AdaptiveNoiseModel
    )
    assert OptimizationObjective.MINIMIZE_DEPTH
    assert CircuitMetrics
    assert QuantumMLPredictor
    

def test_quantum_circuit_optimizer():
    """Test quantum circuit optimization functionality."""
    from quantum_devops_ci.ml_optimization import (
        QuantumCircuitOptimizer, OptimizationObjective, CircuitMetrics
    )
    
    optimizer = QuantumCircuitOptimizer()
    
    # Mock circuit object
    class MockCircuit:
        def depth(self):
            return 20
        def count_ops(self):
            return {'total': 50}
    
    mock_circuit = MockCircuit()
    
    # Test optimization
    result = optimizer.optimize_circuit(
        mock_circuit,
        OptimizationObjective.MINIMIZE_DEPTH,
        technique='gradient_descent',
        max_iterations=10
    )
    
    assert result.original_metrics.depth > 0
    assert result.optimized_metrics.depth >= 0
    assert result.confidence_score > 0
    assert 'depth_reduction' in result.improvements


def test_ml_predictor_training():
    """Test ML predictor training and prediction."""
    from quantum_devops_ci.ml_optimization import QuantumMLPredictor, CircuitMetrics
    
    predictor = QuantumMLPredictor()
    
    # Add training data
    for i in range(20):
        metrics = CircuitMetrics(
            depth=10 + i,
            gate_count=30 + i * 2,
            two_qubit_gates=5 + i,
            estimated_error_rate=0.01 + i * 0.001,
            estimated_execution_time=1.0 + i * 0.1,
            estimated_cost=0.1 + i * 0.01
        )
        predictor.add_training_data(metrics, 'execution_time', 1.0 + i * 0.1)
    
    # Train models
    scores = predictor.train_models(['execution_time'])
    assert 'execution_time' in scores
    assert scores['execution_time'] >= 0
    
    # Test prediction
    test_metrics = CircuitMetrics(
        depth=15, gate_count=40, two_qubit_gates=8,
        estimated_error_rate=0.015, estimated_execution_time=1.5,
        estimated_cost=0.15
    )
    
    prediction, confidence = predictor.predict(test_metrics, 'execution_time')
    assert prediction > 0
    assert 0 <= confidence <= 1


def test_adaptive_noise_model():
    """Test adaptive noise model functionality."""
    from quantum_devops_ci.ml_optimization import AdaptiveNoiseModel
    
    noise_model = AdaptiveNoiseModel()
    
    # Test calibration update
    calibration_data = {
        'gate_error': 0.001,
        'measurement_error': 0.02,
        'thermal_relaxation_time': 100e-6,
        'dephasing_time': 50e-6
    }
    
    noise_model.update_from_calibration('test_backend', calibration_data)
    
    # Test getting current model
    current_model = noise_model.get_current_noise_model('test_backend')
    assert 'gate_error' in current_model
    assert current_model['gate_error'] == 0.001
    
    # Test future prediction
    future_model = noise_model.predict_future_noise('test_backend', 24)
    assert 'gate_error' in future_model
    assert future_model['gate_error'] >= 0


def test_quantum_error_correction_imports():
    """Test quantum error correction module imports."""
    from quantum_devops_ci.quantum_error_correction import (
        QECCode,
        ErrorType,
        ErrorSyndrome,
        LogicalQubit,
        QECResult,
        SurfaceCodeDecoder,
        QuantumErrorCorrection,
        QECBenchmark
    )
    assert QECCode.SURFACE_CODE
    assert ErrorType.BIT_FLIP
    assert LogicalQubit
    assert QuantumErrorCorrection


def test_surface_code_decoder():
    """Test surface code decoder functionality."""
    from quantum_devops_ci.quantum_error_correction import (
        SurfaceCodeDecoder, ErrorSyndrome, ErrorType
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
    assert len(errors) >= 0
    
    if errors:
        qubit_idx, error_type = errors[0]
        assert isinstance(qubit_idx, int)
        assert isinstance(error_type, ErrorType)


def test_quantum_error_correction():
    """Test quantum error correction system."""
    from quantum_devops_ci.quantum_error_correction import (
        QuantumErrorCorrection, QECCode, ErrorSyndrome
    )
    
    qec_system = QuantumErrorCorrection()
    
    # Create logical qubit
    logical_qubit = qec_system.create_logical_qubit(QECCode.SURFACE_CODE, distance=3)
    
    assert logical_qubit.code_type == QECCode.SURFACE_CODE
    assert logical_qubit.distance == 3
    assert len(logical_qubit.physical_qubits) > 0
    assert len(logical_qubit.data_qubits) > 0
    
    # Test stabilizer measurement
    syndrome = qec_system.measure_stabilizers(logical_qubit)
    
    assert isinstance(syndrome.syndrome_bits, list)
    assert len(syndrome.syndrome_bits) > 0
    
    # Test error correction
    qec_result = qec_system.correct_errors(logical_qubit, syndrome)
    
    assert isinstance(qec_result.corrected_errors, list)
    assert qec_result.logical_error_probability >= 0
    assert qec_result.physical_error_rate >= 0


def test_predictive_analytics_imports():
    """Test predictive analytics module imports."""
    from quantum_devops_ci.predictive_analytics import (
        PredictionHorizon,
        MetricType,
        TimeSeriesData,
        PredictionRequest,
        PredictionResult,
        QuantumCostPredictor,
        PerformancePredictor,
        ResourcePlanningEngine
    )
    assert PredictionHorizon.HOUR
    assert MetricType.COST
    assert QuantumCostPredictor
    assert ResourcePlanningEngine


def test_quantum_cost_predictor():
    """Test quantum cost prediction functionality."""
    from quantum_devops_ci.predictive_analytics import QuantumCostPredictor
    
    predictor = QuantumCostPredictor()
    
    # Add historical data
    for i in range(50):
        timestamp = datetime.now() - timedelta(days=i)
        cost = 10.0 + i * 0.1
        features = {
            'shots': 1000 + i * 10,
            'circuit_depth': 10 + i,
            'gate_count': 50 + i * 2,
            'priority_multiplier': 1.0
        }
        predictor.add_historical_data('ibmq', 'qasm_simulator', timestamp, cost, features)
    
    # Train models
    scores = predictor.train_cost_models('ibmq', 'qasm_simulator')
    assert isinstance(scores, dict)
    assert len(scores) > 0
    
    # Test prediction
    result = predictor.predict_cost(
        provider='ibmq',
        backend='qasm_simulator',
        shots=2000,
        circuit_depth=15,
        gate_count=75
    )
    
    assert result.predicted_value > 0
    assert result.confidence_intervals is not None
    assert result.model_confidence > 0


def test_performance_predictor():
    """Test performance prediction functionality."""
    from quantum_devops_ci.predictive_analytics import PerformancePredictor, PredictionHorizon
    
    predictor = PerformancePredictor()
    
    # Add performance data
    for i in range(30):
        job_config = {
            'provider': 'ibmq',
            'backend': 'qasm_simulator',
            'shots': 1000 + i * 100,
            'circuit_depth': 10 + i,
            'gate_count': 50 + i * 5
        }
        predictor.add_performance_data(
            job_config=job_config,
            execution_time=60.0 + i * 2,
            queue_time=300.0 + i * 10,
            success_rate=0.95 - i * 0.001
        )
    
    # Test prediction
    job_config = {
        'provider': 'ibmq',
        'backend': 'qasm_simulator', 
        'shots': 1500,
        'circuit_depth': 20,
        'gate_count': 100
    }
    
    predictions = predictor.predict_performance(job_config, PredictionHorizon.HOUR)
    
    assert 'execution_time' in predictions
    assert 'queue_time' in predictions
    assert 'success_rate' in predictions
    
    for metric, result in predictions.items():
        assert result.predicted_value >= 0
        assert result.model_confidence > 0


def test_resource_planning_engine():
    """Test resource planning functionality."""
    from quantum_devops_ci.predictive_analytics import (
        ResourcePlanningEngine, QuantumCostPredictor, PerformancePredictor
    )
    
    cost_predictor = QuantumCostPredictor()
    performance_predictor = PerformancePredictor()
    
    # Add minimal data for predictors
    timestamp = datetime.now()
    cost_predictor.add_historical_data('ibmq', 'qasm_simulator', timestamp, 10.0, {
        'shots': 1000, 'circuit_depth': 10, 'gate_count': 50, 'priority_multiplier': 1.0
    })
    
    planning_engine = ResourcePlanningEngine(cost_predictor, performance_predictor)
    
    # Test resource planning
    experiments = [
        {'id': 'exp1', 'shots': 1000, 'circuit_depth': 10, 'gate_count': 50},
        {'id': 'exp2', 'shots': 2000, 'circuit_depth': 15, 'gate_count': 75}
    ]
    
    plan = planning_engine.create_resource_plan(
        experiments=experiments,
        budget_limit=100.0,
        time_limit=timedelta(hours=24),
        optimization_goal='cost'
    )
    
    assert 'assignments' in plan
    assert 'total_estimated_cost' in plan
    assert 'recommendations' in plan
    assert plan['total_experiments'] == 2


def test_advanced_validation_imports():
    """Test advanced validation module imports."""
    from quantum_devops_ci.advanced_validation import (
        ValidationLevel,
        StatisticalTest,
        ValidationMetrics,
        ExperimentalDesign,
        BaseValidator,
        QuantumAlgorithmValidator,
        MLModelValidator,
        ResearchExperimentValidator
    )
    assert ValidationLevel.RESEARCH
    assert StatisticalTest.T_TEST
    assert ValidationMetrics
    assert QuantumAlgorithmValidator


def test_quantum_algorithm_validator():
    """Test quantum algorithm validation."""
    from quantum_devops_ci.advanced_validation import (
        QuantumAlgorithmValidator, ValidationLevel, ExperimentalDesign
    )
    
    validator = QuantumAlgorithmValidator(ValidationLevel.RESEARCH)
    
    # Mock algorithm results
    algorithm_results = {
        'algorithm_name': 'test_algorithm',
        'predictions': [1.0, 2.0, 3.0, 4.0, 5.0] * 10,  # 50 predictions
        'ground_truth': [1.1, 1.9, 3.1, 3.9, 5.1] * 10   # 50 ground truth values
    }
    
    # Test basic validation
    metrics = validator.validate(algorithm_results)
    
    assert metrics.accuracy >= 0
    assert metrics.precision >= 0
    assert metrics.recall >= 0
    assert metrics.f1_score >= 0
    
    # Test with experimental design
    experimental_design = ExperimentalDesign(
        hypothesis="Algorithm performs better than baseline",
        null_hypothesis="Algorithm performs same as baseline",
        alternative_hypothesis="Algorithm performs better than baseline",
        significance_level=0.05,
        power_target=0.8
    )
    
    metrics_with_design = validator.validate(algorithm_results, experimental_design=experimental_design)
    assert metrics_with_design.p_value is not None


def test_ml_model_validator():
    """Test ML model validation."""
    from quantum_devops_ci.advanced_validation import MLModelValidator, ValidationLevel
    import numpy as np
    
    validator = MLModelValidator(ValidationLevel.STANDARD)
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return X[:, 0] + 0.1 * np.random.randn(len(X))
        
        def score(self, X, y):
            predictions = self.predict(X)
            return 1 - np.mean((y - predictions) ** 2) / np.var(y)
    
    model = MockModel()
    
    # Generate test data
    X_test = np.random.randn(50, 3)
    y_test = X_test[:, 0] + 0.1 * np.random.randn(50)
    
    # Test validation
    metrics = validator.validate(model, X_test, y_test)
    
    assert metrics.accuracy >= 0
    assert metrics.precision >= 0
    assert metrics.f1_score >= 0


def test_research_experiment_validator():
    """Test research experiment validation."""
    from quantum_devops_ci.advanced_validation import ResearchExperimentValidator
    
    validator = ResearchExperimentValidator()
    
    # Mock experimental data
    experimental_data = {
        'measurements': [1.0, 1.1, 0.9, 1.05, 0.95] * 10  # 50 measurements
    }
    
    control_data = {
        'measurements': [0.8, 0.9, 0.85, 0.88, 0.82] * 10  # 50 control measurements
    }
    
    # Test experiment validation
    report = validator.validate_experiment(
        experiment_id='test_exp_001',
        hypothesis='Treatment increases measurement values',
        experimental_data=experimental_data,
        control_data=control_data
    )
    
    assert 'experiment_id' in report
    assert 'results' in report
    assert 'statistical_tests' in report
    assert 'publication_readiness' in report
    assert report['results']['treatment']['n'] == 50
    assert report['results']['control']['n'] == 50


def test_quantum_sovereignty_imports():
    """Test quantum sovereignty module imports."""
    from quantum_devops_ci.quantum_sovereignty import (
        SovereigntyLevel,
        TechnologyClassification,
        ExportControlRegime,
        SovereigntyPolicy,
        AccessRequest,
        ComplianceReport,
        QuantumSovereigntyManager,
        QuantumDataSovereignty
    )
    assert SovereigntyLevel.CONTROLLED
    assert TechnologyClassification.DUAL_USE
    assert QuantumSovereigntyManager
    assert QuantumDataSovereignty


def test_quantum_sovereignty_manager():
    """Test quantum sovereignty management."""
    from quantum_devops_ci.quantum_sovereignty import (
        QuantumSovereigntyManager, AccessRequest, TechnologyClassification
    )
    
    manager = QuantumSovereigntyManager()
    
    # Test technology classification
    classification = manager.classify_quantum_technology(
        algorithm_description='Quantum optimization for logistics',
        intended_application='Supply chain optimization',
        technical_parameters={'qubits': 50, 'gate_fidelity': 0.99}
    )
    
    assert classification in [
        TechnologyClassification.DUAL_USE,
        TechnologyClassification.APPLIED_RESEARCH,
        TechnologyClassification.COMMERCIAL
    ]
    
    # Test access request evaluation
    request = AccessRequest(
        request_id='req_001',
        requester_info={'country': 'US', 'organization': 'University'},
        requested_technology=TechnologyClassification.DUAL_USE,
        intended_use='Research collaboration',
        destination_country='CA',
        duration=timedelta(days=365),
        justification='Academic research project'
    )
    
    evaluation = manager.evaluate_access_request(request)
    
    assert 'request_id' in evaluation
    assert 'approved' in evaluation
    assert 'risk_score' in evaluation
    assert 'reasoning' in evaluation


def test_quantum_data_sovereignty():
    """Test quantum data sovereignty functionality."""
    from quantum_devops_ci.quantum_sovereignty import (
        QuantumSovereigntyManager, QuantumDataSovereignty
    )
    
    sovereignty_manager = QuantumSovereigntyManager()
    data_sovereignty = QuantumDataSovereignty(sovereignty_manager)
    
    # Test data transfer assessment
    assessment = data_sovereignty.assess_data_transfer(
        data_type='research_data',
        source_country='US',
        destination_country='CA',
        data_classification='SENSITIVE',
        transfer_volume_gb=10.5
    )
    
    assert 'transfer_id' in assessment
    assert 'approved' in assessment
    assert 'requirements' in assessment
    assert 'encryption_required' in assessment
    
    # Test encryption requirements
    encryption_req = data_sovereignty.get_encryption_requirements(
        source_country='US',
        destination_country='CN',
        data_classification='CLASSIFIED'
    )
    
    assert 'minimum_algorithm' in encryption_req
    assert 'quantum_safe' in encryption_req
    assert encryption_req['quantum_safe']  # Should require quantum-safe encryption for classified data


def test_cross_border_collaboration_monitoring():
    """Test cross-border collaboration monitoring."""
    from quantum_devops_ci.quantum_sovereignty import QuantumSovereigntyManager
    
    manager = QuantumSovereigntyManager()
    
    participants = [
        {'name': 'Alice', 'country': 'US', 'organization': 'MIT'},
        {'name': 'Bob', 'country': 'CA', 'organization': 'University of Toronto'},
        {'name': 'Charlie', 'country': 'GB', 'organization': 'Oxford'}
    ]
    
    data_sharing_plan = {
        'data_location': 'US',
        'data_types': ['research_results', 'measurement_data'],
        'retention_period': '5_years'
    }
    
    report = manager.monitor_cross_border_collaboration(
        participants=participants,
        project_description='Quantum algorithm development for optimization',
        data_sharing_plan=data_sharing_plan
    )
    
    assert 'collaboration_id' in report
    assert 'compliance_status' in report
    assert 'participants' in report
    assert report['compliance_status'] in ['COMPLIANT', 'REVIEW_REQUIRED', 'VIOLATION']


def run_all_tests():
    """Run all Generation 4 validation tests."""
    print("🚀 Starting Generation 4 Intelligence Validation Suite")
    print("=" * 60)
    
    # ML Optimization Tests
    print("\n📊 ML Optimization Module Tests:")
    run_test("ML Optimization Imports", test_ml_optimization_imports)
    run_test("Quantum Circuit Optimizer", test_quantum_circuit_optimizer)
    run_test("ML Predictor Training", test_ml_predictor_training)
    run_test("Adaptive Noise Model", test_adaptive_noise_model)
    
    # Quantum Error Correction Tests
    print("\n🔧 Quantum Error Correction Tests:")
    run_test("QEC Module Imports", test_quantum_error_correction_imports)
    run_test("Surface Code Decoder", test_surface_code_decoder)
    run_test("Quantum Error Correction System", test_quantum_error_correction)
    
    # Predictive Analytics Tests
    print("\n📈 Predictive Analytics Tests:")
    run_test("Predictive Analytics Imports", test_predictive_analytics_imports)
    run_test("Quantum Cost Predictor", test_quantum_cost_predictor)
    run_test("Performance Predictor", test_performance_predictor)
    run_test("Resource Planning Engine", test_resource_planning_engine)
    
    # Advanced Validation Tests
    print("\n🧪 Advanced Validation Tests:")
    run_test("Advanced Validation Imports", test_advanced_validation_imports)
    run_test("Quantum Algorithm Validator", test_quantum_algorithm_validator)
    run_test("ML Model Validator", test_ml_model_validator)
    run_test("Research Experiment Validator", test_research_experiment_validator)
    
    # Quantum Sovereignty Tests
    print("\n🌐 Quantum Sovereignty Tests:")
    run_test("Quantum Sovereignty Imports", test_quantum_sovereignty_imports)
    run_test("Quantum Sovereignty Manager", test_quantum_sovereignty_manager)
    run_test("Quantum Data Sovereignty", test_quantum_data_sovereignty)
    run_test("Cross-Border Collaboration", test_cross_border_collaboration_monitoring)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 GENERATION 4 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']} ✅")
    print(f"Failed: {test_results['failed_tests']} ❌")
    
    success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Generation 4 features are production-ready!")
    elif success_rate >= 80:
        print("🟢 GOOD: Generation 4 features are mostly functional")  
    elif success_rate >= 70:
        print("🟡 ACCEPTABLE: Generation 4 features need minor fixes")
    else:
        print("🔴 NEEDS WORK: Generation 4 features require significant improvements")
        
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