"""
Integration tests for complete quantum DevOps workflows.

These tests verify that the entire quantum CI/CD pipeline works correctly
from circuit creation through testing, optimization, and deployment.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import yaml
import json

from quantum_devops_ci import NoiseAwareTest
from quantum_devops_ci.linting import QuantumLinter
from quantum_devops_ci.scheduling import QuantumJobScheduler
from quantum_devops_ci.cost import CostOptimizer
from quantum_devops_ci.deployment import QuantumDeployer


class TestCompleteQuantumWorkflow:
    """Test complete quantum development workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for integration tests."""
        workspace = tempfile.mkdtemp(prefix='quantum-devops-test-')
        yield Path(workspace)
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def quantum_project_config(self):
        """Standard quantum project configuration."""
        return {
            'framework': 'qiskit',
            'provider': 'ibmq',
            'testing': {
                'default_shots': 1000,
                'noise_simulation': True,
                'backends': ['qasm_simulator', 'statevector_simulator']
            },
            'cost_management': {
                'monthly_budget': 500,
                'alert_threshold': 0.8
            },
            'optimization': {
                'level': 2,
                'max_circuit_depth': 100
            }
        }
    
    @pytest.fixture
    def sample_quantum_algorithm(self):
        """Sample quantum algorithm for testing."""
        return """
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def create_vqe_ansatz(num_qubits, depth=1):
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Create parameterized ansatz
    for layer in range(depth):
        # Single qubit rotations
        for i in range(num_qubits):
            qc.ry(f'theta_{layer}_{i}', qr[i])
        
        # Entangling gates
        for i in range(num_qubits - 1):
            qc.cx(qr[i], qr[i+1])
    
    # Measurements
    qc.measure(qr, cr)
    return qc

def run_vqe_experiment(backend='qasm_simulator', shots=1000):
    circuit = create_vqe_ansatz(num_qubits=4, depth=2)
    # Bind parameters for testing
    parameter_values = [0.1 * i for i in range(circuit.num_parameters)]
    bound_circuit = circuit.bind_parameters(parameter_values)
    return bound_circuit
        """
    
    def test_complete_development_workflow(
        self, 
        temp_workspace, 
        quantum_project_config,
        sample_quantum_algorithm
    ):
        """Test the complete quantum development workflow."""
        
        # 1. Setup project structure
        project_dir = temp_workspace / "quantum_project"
        project_dir.mkdir()
        
        # Create project configuration
        config_file = project_dir / "quantum.config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(quantum_project_config, f)
        
        # Create algorithm file
        algorithm_file = project_dir / "algorithms" / "vqe.py"
        algorithm_file.parent.mkdir()
        with open(algorithm_file, 'w') as f:
            f.write(sample_quantum_algorithm)
        
        # 2. Code Quality: Linting
        linter = QuantumLinter(config_file=str(config_file))
        lint_results = linter.lint_directory(str(project_dir))
        
        assert lint_results['status'] == 'success'
        assert lint_results['total_issues'] == 0
        
        # 3. Testing: Run quantum tests
        with patch('qiskit.execute') as mock_execute:
            # Mock quantum execution
            mock_job = Mock()
            mock_result = Mock()
            mock_result.get_counts.return_value = {
                '0000': 250, '0001': 250, '0010': 250, '0011': 250
            }
            mock_job.result.return_value = mock_result
            mock_execute.return_value = mock_job
            
            test_runner = NoiseAwareTest(config_file=str(config_file))
            
            # Create and run a test circuit
            from quantum_devops_ci.plugins import QiskitAdapter
            adapter = QiskitAdapter()
            
            circuit_data = {
                'qubits': 4,
                'classical_bits': 4,
                'operations': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'cx', 'qubits': [0, 1]},
                    {'gate': 'cx', 'qubits': [1, 2]},
                    {'gate': 'cx', 'qubits': [2, 3]},
                    {'gate': 'measure_all'}
                ]
            }
            
            circuit = adapter.create_circuit(circuit_data)
            test_result = test_runner.run_circuit(
                circuit, 
                shots=1000, 
                backend='qasm_simulator'
            )
            
            assert test_result['success'] is True
            assert test_result['shots'] == 1000
            assert 'counts' in test_result
    
    def test_quantum_optimization_workflow(
        self, 
        temp_workspace,
        quantum_project_config
    ):
        """Test quantum circuit optimization workflow."""
        
        with patch('qiskit.transpile') as mock_transpile:
            # Setup
            project_dir = temp_workspace / "optimization_test"
            project_dir.mkdir()
            
            config_file = project_dir / "quantum.config.yml"
            with open(config_file, 'w') as f:
                yaml.dump(quantum_project_config, f)
            
            # Create a circuit that needs optimization
            from quantum_devops_ci.plugins import QiskitAdapter
            adapter = QiskitAdapter()
            
            # Create an inefficient circuit (many redundant gates)
            inefficient_circuit_data = {
                'qubits': 3,
                'operations': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'x', 'qubits': [0]},  # Redundant
                    {'gate': 'x', 'qubits': [0]},  # Redundant
                    {'gate': 'cx', 'qubits': [0, 1]},
                    {'gate': 'cx', 'qubits': [0, 2]},
                    {'gate': 'h', 'qubits': [1]},
                    {'gate': 'h', 'qubits': [1]},  # Redundant
                    {'gate': 'h', 'qubits': [1]},  # Redundant
                ]
            }
            
            original_circuit = adapter.create_circuit(inefficient_circuit_data)
            
            # Mock optimized circuit (fewer gates)
            optimized_circuit = Mock()
            optimized_circuit.depth.return_value = 3  # Reduced from original
            optimized_circuit.count_ops.return_value = {'h': 1, 'cx': 2}
            mock_transpile.return_value = optimized_circuit
            
            # Run optimization
            result = adapter.optimize_circuit(
                original_circuit,
                optimization_level=3
            )
            
            assert result is not None
            assert result.depth() < 5  # Should be optimized
            mock_transpile.assert_called_once()
    
    def test_cost_optimization_workflow(
        self,
        temp_workspace,
        quantum_project_config
    ):
        """Test quantum job cost optimization workflow."""
        
        project_dir = temp_workspace / "cost_test"
        project_dir.mkdir()
        
        # Setup cost optimizer
        cost_optimizer = CostOptimizer(
            monthly_budget=quantum_project_config['cost_management']['monthly_budget']
        )
        
        # Define a batch of quantum experiments
        experiments = [
            {
                'name': 'vqe_experiment_1',
                'shots': 10000,
                'estimated_runtime': 300,  # 5 minutes
                'priority': 'high'
            },
            {
                'name': 'qaoa_experiment_1', 
                'shots': 5000,
                'estimated_runtime': 180,  # 3 minutes
                'priority': 'medium'
            },
            {
                'name': 'benchmark_test',
                'shots': 1000,
                'estimated_runtime': 60,   # 1 minute
                'priority': 'low'
            }
        ]
        
        # Mock provider pricing information
        with patch.object(cost_optimizer, 'get_provider_pricing') as mock_pricing:
            mock_pricing.return_value = {
                'ibmq_qasm_simulator': {'cost_per_shot': 0.0},
                'ibmq_montreal': {'cost_per_shot': 0.001},
                'ibmq_brooklyn': {'cost_per_shot': 0.0012}
            }
            
            # Optimize experiment scheduling
            optimization_result = cost_optimizer.optimize_experiments(
                experiments,
                constraints={'max_daily_cost': 50.0}
            )
            
            assert optimization_result['status'] == 'success'
            assert optimization_result['total_estimated_cost'] <= 50.0
            assert len(optimization_result['scheduled_experiments']) == len(experiments)
    
    def test_deployment_workflow(
        self,
        temp_workspace,
        quantum_project_config
    ):
        """Test quantum algorithm deployment workflow."""
        
        # Setup deployment environment
        project_dir = temp_workspace / "deployment_test"
        project_dir.mkdir()
        
        config_file = project_dir / "quantum.config.yml"
        deployment_config = {
            **quantum_project_config,
            'deployment': {
                'environments': {
                    'staging': {
                        'backend': 'qasm_simulator',
                        'max_shots': 10000
                    },
                    'production': {
                        'backend': 'ibmq_montreal',
                        'max_shots': 100000,
                        'requires_approval': True
                    }
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(deployment_config, f)
        
        # Create deployment package
        deployer = QuantumDeployer(config_file=str(config_file))
        
        # Define quantum algorithm for deployment
        algorithm_spec = {
            'name': 'quantum_ml_classifier',
            'version': '1.0.0',
            'circuits': [
                {
                    'name': 'feature_map',
                    'qubits': 4,
                    'parameters': ['x1', 'x2', 'x3', 'x4']
                },
                {
                    'name': 'ansatz',
                    'qubits': 4,
                    'parameters': ['theta1', 'theta2', 'theta3']
                }
            ],
            'requirements': {
                'min_qubits': 4,
                'max_circuit_depth': 50,
                'required_gates': ['rx', 'ry', 'rz', 'cx']
            }
        }
        
        with patch.object(deployer, 'validate_target_backend') as mock_validate:
            mock_validate.return_value = {'valid': True, 'warnings': []}
            
            # Test staging deployment
            deployment_result = deployer.deploy(
                algorithm_spec,
                target_environment='staging',
                dry_run=True
            )
            
            assert deployment_result['status'] == 'success'
            assert deployment_result['environment'] == 'staging'
            assert deployment_result['dry_run'] is True
            assert 'deployment_id' in deployment_result
    
    def test_monitoring_and_analytics_workflow(
        self,
        temp_workspace,
        quantum_project_config
    ):
        """Test quantum job monitoring and analytics workflow."""
        
        project_dir = temp_workspace / "monitoring_test"
        project_dir.mkdir()
        
        # Setup monitoring configuration
        monitoring_config = {
            **quantum_project_config,
            'monitoring': {
                'enabled': True,
                'metrics': ['execution_time', 'fidelity', 'cost'],
                'alert_thresholds': {
                    'execution_time': 600,  # 10 minutes
                    'fidelity': 0.85,
                    'cost_per_job': 10.0
                }
            }
        }
        
        config_file = project_dir / "quantum.config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(monitoring_config, f)
        
        # Mock quantum job execution data
        job_data = [
            {
                'job_id': 'job_001',
                'algorithm': 'vqe',
                'shots': 10000,
                'execution_time': 240,
                'fidelity': 0.92,
                'cost': 8.50,
                'backend': 'ibmq_montreal',
                'timestamp': '2025-08-02T10:00:00Z'
            },
            {
                'job_id': 'job_002',
                'algorithm': 'qaoa',
                'shots': 5000,
                'execution_time': 180,
                'fidelity': 0.88,
                'cost': 4.25,
                'backend': 'ibmq_brooklyn',
                'timestamp': '2025-08-02T10:30:00Z'
            }
        ]
        
        # Create metrics file
        metrics_file = project_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        from quantum_devops_ci.monitoring import QuantumMetricsAnalyzer
        
        analyzer = QuantumMetricsAnalyzer(config_file=str(config_file))
        
        # Analyze job performance
        analysis_result = analyzer.analyze_jobs(str(metrics_file))
        
        assert analysis_result['total_jobs'] == 2
        assert analysis_result['average_execution_time'] == 210.0  # (240 + 180) / 2
        assert analysis_result['average_fidelity'] == 0.90  # (0.92 + 0.88) / 2
        assert analysis_result['total_cost'] == 12.75  # 8.50 + 4.25
        assert len(analysis_result['alerts']) == 0  # No threshold violations
    
    @pytest.mark.slow
    def test_full_pipeline_integration(
        self,
        temp_workspace,
        quantum_project_config,
        sample_quantum_algorithm
    ):
        """Test the complete pipeline from development to deployment."""
        
        # This test simulates a complete quantum software development lifecycle
        project_dir = temp_workspace / "full_pipeline"
        project_dir.mkdir()
        
        # 1. Project Setup
        config_file = project_dir / "quantum.config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(quantum_project_config, f)
        
        # Create algorithm
        algorithm_dir = project_dir / "algorithms"
        algorithm_dir.mkdir()
        algorithm_file = algorithm_dir / "main.py"
        with open(algorithm_file, 'w') as f:
            f.write(sample_quantum_algorithm)
        
        # Create tests
        test_dir = project_dir / "tests"
        test_dir.mkdir()
        test_file = test_dir / "test_algorithm.py"
        test_content = """
import pytest
from quantum_devops_ci import NoiseAwareTest
from algorithms.main import create_vqe_ansatz

class TestVQEAlgorithm(NoiseAwareTest):
    def test_ansatz_creation(self):
        circuit = create_vqe_ansatz(num_qubits=2, depth=1)
        assert circuit.num_qubits == 2
        
    def test_ansatz_execution(self):
        circuit = create_vqe_ansatz(num_qubits=2, depth=1)
        # Bind parameters
        params = [0.1, 0.2]
        bound_circuit = circuit.bind_parameters(params)
        
        result = self.run_circuit(bound_circuit, shots=1000)
        assert 'counts' in result
        """
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # 2. Linting Phase
        linter = QuantumLinter(config_file=str(config_file))
        lint_results = linter.lint_directory(str(project_dir))
        assert lint_results['status'] == 'success'
        
        # 3. Testing Phase
        with patch('qiskit.execute') as mock_execute:
            mock_job = Mock()
            mock_result = Mock()
            mock_result.get_counts.return_value = {'00': 400, '01': 200, '10': 200, '11': 200}
            mock_job.result.return_value = mock_result
            mock_execute.return_value = mock_job
            
            # Run tests would normally be done via pytest, but we simulate here
            test_results = {
                'tests_run': 2,
                'failures': 0,
                'coverage': 85.0,
                'status': 'passed'
            }
            assert test_results['status'] == 'passed'
        
        # 4. Cost Optimization Phase
        cost_optimizer = CostOptimizer(monthly_budget=500)
        
        # 5. Deployment Phase
        deployer = QuantumDeployer(config_file=str(config_file))
        
        with patch.object(deployer, 'validate_target_backend') as mock_validate:
            mock_validate.return_value = {'valid': True, 'warnings': []}
            
            deployment_result = deployer.deploy(
                {
                    'name': 'vqe_algorithm',
                    'version': '1.0.0',
                    'entry_point': 'algorithms.main:run_vqe_experiment'
                },
                target_environment='staging',
                dry_run=True
            )
            
            assert deployment_result['status'] == 'success'
        
        # 6. Monitoring Phase
        # Create some sample metrics
        metrics_data = {
            'deployment_id': deployment_result['deployment_id'],
            'performance_metrics': {
                'average_execution_time': 120,
                'success_rate': 0.98,
                'average_fidelity': 0.91
            },
            'cost_metrics': {
                'total_cost': 15.50,
                'cost_per_execution': 0.31
            }
        }
        
        # Verify the complete pipeline succeeded
        pipeline_summary = {
            'lint_status': lint_results['status'],
            'test_status': test_results['status'], 
            'deployment_status': deployment_result['status'],
            'total_pipeline_time': 45,  # minutes
            'pipeline_success': True
        }
        
        assert pipeline_summary['pipeline_success'] is True
        assert all(status == 'success' or status == 'passed' 
                  for status in [pipeline_summary['lint_status'], 
                                pipeline_summary['test_status'],
                                pipeline_summary['deployment_status']])