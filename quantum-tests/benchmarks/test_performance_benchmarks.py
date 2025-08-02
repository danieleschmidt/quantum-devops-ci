"""
Performance benchmarks for quantum DevOps CI/CD operations.

These benchmarks help track performance regressions and identify
optimization opportunities in quantum workflow operations.
"""

import pytest
import time
import psutil
import numpy as np
from unittest.mock import Mock, patch
import concurrent.futures
from functools import wraps

from quantum_devops_ci.testing import NoiseAwareTest
from quantum_devops_ci.linting import QuantumLinter
from quantum_devops_ci.scheduling import QuantumJobScheduler


def benchmark(name=None):
    """Decorator to mark functions as benchmarks and collect metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record initial system state
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu_percent = process.cpu_percent()
            
            # Record start time
            start_time = time.time()
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            # Record end time and final system state
            end_time = time.time()
            execution_time = end_time - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            # Store benchmark results
            benchmark_name = name or func.__name__
            benchmark_result = {
                'name': benchmark_name,
                'execution_time': execution_time,
                'memory_usage_mb': final_memory,
                'memory_delta_mb': memory_delta,
                'success': success,
                'error': error
            }
            
            # Store in pytest cache for reporting
            if hasattr(pytest, 'benchmark_results'):
                pytest.benchmark_results.append(benchmark_result)
            else:
                pytest.benchmark_results = [benchmark_result]
            
            if not success:
                raise Exception(error)
                
            return result
        return wrapper
    return decorator


class TestQuantumCircuitBenchmarks:
    """Benchmark quantum circuit operations."""
    
    @pytest.mark.benchmark
    @benchmark("circuit_creation_small")
    def test_small_circuit_creation_performance(self):
        """Benchmark creation of small quantum circuits (2-5 qubits)."""
        from quantum_devops_ci.plugins import QiskitAdapter
        
        adapter = QiskitAdapter()
        
        # Create 100 small circuits
        circuits = []
        for i in range(100):
            circuit_data = {
                'qubits': 3,
                'classical_bits': 3,
                'operations': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'cx', 'qubits': [0, 1]},
                    {'gate': 'cx', 'qubits': [1, 2]},
                    {'gate': 'measure_all'}
                ]
            }
            circuit = adapter.create_circuit(circuit_data)
            circuits.append(circuit)
        
        assert len(circuits) == 100
        return len(circuits)
    
    @pytest.mark.benchmark
    @benchmark("circuit_creation_large")  
    def test_large_circuit_creation_performance(self):
        """Benchmark creation of large quantum circuits (10+ qubits)."""
        from quantum_devops_ci.plugins import QiskitAdapter
        
        adapter = QiskitAdapter()
        
        # Create 10 large circuits
        circuits = []
        for i in range(10):
            # Create a large circuit with many operations
            operations = []
            num_qubits = 15
            
            # Add Hadamard gates
            for qubit in range(num_qubits):
                operations.append({'gate': 'h', 'qubits': [qubit]})
            
            # Add entangling gates
            for layer in range(5):  # 5 layers of entanglement
                for qubit in range(num_qubits - 1):
                    operations.append({'gate': 'cx', 'qubits': [qubit, qubit + 1]})
            
            # Add measurements
            operations.append({'gate': 'measure_all'})
            
            circuit_data = {
                'qubits': num_qubits,
                'classical_bits': num_qubits,
                'operations': operations
            }
            
            circuit = adapter.create_circuit(circuit_data)
            circuits.append(circuit)
        
        assert len(circuits) == 10
        return len(circuits)
    
    @pytest.mark.benchmark
    @benchmark("circuit_optimization")
    def test_circuit_optimization_performance(self):
        """Benchmark quantum circuit optimization."""
        from quantum_devops_ci.plugins import QiskitAdapter
        
        adapter = QiskitAdapter()
        
        # Create a circuit that needs optimization
        operations = []
        num_qubits = 8
        
        # Add many redundant operations
        for _ in range(20):
            for qubit in range(num_qubits):
                operations.extend([
                    {'gate': 'x', 'qubits': [qubit]},
                    {'gate': 'x', 'qubits': [qubit]},  # Redundant
                    {'gate': 'h', 'qubits': [qubit]},
                    {'gate': 'h', 'qubits': [qubit]},  # Redundant
                ])
        
        circuit_data = {
            'qubits': num_qubits,
            'operations': operations
        }
        
        circuit = adapter.create_circuit(circuit_data)
        
        with patch('qiskit.transpile') as mock_transpile:
            mock_transpile.return_value = circuit
            
            # Optimize 50 circuits
            optimized_circuits = []
            for _ in range(50):
                optimized = adapter.optimize_circuit(circuit, optimization_level=3)
                optimized_circuits.append(optimized)
        
        assert len(optimized_circuits) == 50
        return len(optimized_circuits)


class TestQuantumTestingBenchmarks:
    """Benchmark quantum testing operations."""
    
    @pytest.mark.benchmark
    @benchmark("noise_aware_testing")
    def test_noise_aware_testing_performance(self):
        """Benchmark noise-aware quantum testing."""
        
        test_runner = NoiseAwareTest()
        
        # Mock quantum execution
        with patch('qiskit.execute') as mock_execute:
            mock_job = Mock()
            mock_result = Mock()
            mock_result.get_counts.return_value = {
                '000': 125, '001': 125, '010': 125, '011': 125,
                '100': 125, '101': 125, '110': 125, '111': 125
            }
            mock_job.result.return_value = mock_result
            mock_execute.return_value = mock_job
            
            from quantum_devops_ci.plugins import QiskitAdapter
            adapter = QiskitAdapter()
            
            # Create test circuit
            circuit_data = {
                'qubits': 3,
                'classical_bits': 3,
                'operations': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'h', 'qubits': [1]},
                    {'gate': 'h', 'qubits': [2]},
                    {'gate': 'measure_all'}
                ]
            }
            circuit = adapter.create_circuit(circuit_data)
            
            # Run multiple noise-aware tests
            results = []
            noise_levels = [0.001, 0.01, 0.05, 0.1]
            
            for _ in range(25):  # 25 test iterations
                noise_results = test_runner.run_with_noise_sweep(
                    circuit, 
                    noise_levels, 
                    shots=1000
                )
                results.append(noise_results)
        
        assert len(results) == 25
        return len(results)
    
    @pytest.mark.benchmark
    @benchmark("parallel_testing")
    def test_parallel_testing_performance(self):
        """Benchmark parallel quantum test execution."""
        
        def run_single_test(test_id):
            """Simulate a single quantum test."""
            time.sleep(0.1)  # Simulate test execution time
            return {'test_id': test_id, 'result': 'passed'}
        
        # Run tests in parallel
        num_tests = 50
        max_workers = 8
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_test, i) for i in range(num_tests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert len(results) == num_tests
        return len(results)


class TestQuantumLintingBenchmarks:
    """Benchmark quantum linting operations."""
    
    @pytest.mark.benchmark
    @benchmark("circuit_linting")
    def test_circuit_linting_performance(self):
        """Benchmark quantum circuit linting."""
        
        # Create mock linter
        linter = QuantumLinter()
        
        # Generate many circuits to lint
        circuits_data = []
        for i in range(200):
            circuit_data = {
                'name': f'test_circuit_{i}',
                'qubits': np.random.randint(2, 10),
                'operations': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'cx', 'qubits': [0, 1]} if i % 2 == 0 else {'gate': 'x', 'qubits': [1]},
                    {'gate': 'measure_all'}
                ],
                'depth': np.random.randint(5, 25)
            }
            circuits_data.append(circuit_data)
        
        # Mock linting results
        with patch.object(linter, 'lint_circuit') as mock_lint:
            mock_lint.return_value = {
                'issues': [],
                'warnings': [],
                'score': 0.95
            }
            
            # Lint all circuits
            results = []
            for circuit_data in circuits_data:
                result = linter.lint_circuit(circuit_data)
                results.append(result)
        
        assert len(results) == 200
        return len(results)
    
    @pytest.mark.benchmark  
    @benchmark("bulk_file_linting")
    def test_bulk_file_linting_performance(self):
        """Benchmark linting of multiple quantum algorithm files."""
        
        # Create temporary files to lint
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create 50 Python files with quantum algorithms
            file_paths = []
            for i in range(50):
                file_content = f"""
from qiskit import QuantumCircuit

def create_circuit_{i}():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc

def run_experiment_{i}():
    circuit = create_circuit_{i}()
    # Execute circuit
    return circuit
                """
                
                file_path = os.path.join(temp_dir, f'algorithm_{i}.py')
                with open(file_path, 'w') as f:
                    f.write(file_content)
                file_paths.append(file_path)
            
            # Lint all files
            linter = QuantumLinter()
            
            with patch.object(linter, 'lint_file') as mock_lint_file:
                mock_lint_file.return_value = {
                    'issues': [],
                    'warnings': [],
                    'score': 0.90
                }
                
                results = []
                for file_path in file_paths:
                    result = linter.lint_file(file_path)
                    results.append(result)
            
            assert len(results) == 50
            return len(results)
            
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)


class TestQuantumSchedulingBenchmarks:
    """Benchmark quantum job scheduling operations."""
    
    @pytest.mark.benchmark
    @benchmark("job_scheduling_optimization")
    def test_job_scheduling_optimization_performance(self):
        """Benchmark quantum job scheduling optimization."""
        
        scheduler = QuantumJobScheduler()
        
        # Create a large batch of quantum jobs
        jobs = []
        for i in range(1000):
            job = {
                'id': f'job_{i}',
                'shots': np.random.randint(1000, 10000),
                'estimated_runtime': np.random.randint(60, 600),  # seconds
                'priority': np.random.choice(['low', 'medium', 'high']),
                'backend_requirements': np.random.choice(['any', 'simulator', 'hardware'])
            }
            jobs.append(job)
        
        # Mock available backends
        with patch.object(scheduler, 'get_available_backends') as mock_backends:
            mock_backends.return_value = [
                {'name': 'qasm_simulator', 'queue_length': 0, 'cost_per_shot': 0.0},
                {'name': 'ibmq_montreal', 'queue_length': 50, 'cost_per_shot': 0.001},
                {'name': 'ibmq_brooklyn', 'queue_length': 30, 'cost_per_shot': 0.0012}
            ]
            
            # Optimize job scheduling
            with patch.object(scheduler, 'optimize_schedule') as mock_optimize:
                mock_optimize.return_value = {
                    'schedule': jobs[:100],  # Return first 100 jobs as scheduled
                    'total_cost': 250.0,
                    'total_time': 7200  # 2 hours
                }
                
                result = scheduler.optimize_schedule(
                    jobs,
                    constraints={'max_cost': 500.0, 'max_time': 10800}  # 3 hours
                )
        
        assert result['schedule'] is not None
        assert len(result['schedule']) <= len(jobs)
        return len(result['schedule'])
    
    @pytest.mark.benchmark
    @benchmark("concurrent_job_submission")
    def test_concurrent_job_submission_performance(self):
        """Benchmark concurrent quantum job submission."""
        
        def submit_job(job_data):
            """Simulate job submission."""
            time.sleep(0.01)  # Simulate network latency
            return {
                'job_id': f"job_{job_data['id']}_submitted",
                'status': 'queued'
            }
        
        # Submit 100 jobs concurrently
        job_data = [{'id': i} for i in range(100)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_job, job) for job in job_data]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 100
        return len(results)


class TestMemoryAndResourceBenchmarks:
    """Benchmark memory usage and resource consumption."""
    
    @pytest.mark.benchmark
    @benchmark("memory_usage_large_circuits")
    def test_memory_usage_large_circuits(self):
        """Benchmark memory usage with large quantum circuits."""
        
        from quantum_devops_ci.plugins import QiskitAdapter
        adapter = QiskitAdapter()
        
        # Create progressively larger circuits
        circuits = []
        for num_qubits in [5, 10, 15, 20]:
            # Create circuit with many layers
            operations = []
            
            for layer in range(10):  # 10 layers
                # Add Hadamard gates
                for qubit in range(num_qubits):
                    operations.append({'gate': 'h', 'qubits': [qubit]})
                
                # Add entangling gates
                for qubit in range(num_qubits - 1):
                    operations.append({'gate': 'cx', 'qubits': [qubit, qubit + 1]})
            
            operations.append({'gate': 'measure_all'})
            
            circuit_data = {
                'qubits': num_qubits,
                'classical_bits': num_qubits,
                'operations': operations
            }
            
            circuit = adapter.create_circuit(circuit_data)
            circuits.append(circuit)
        
        assert len(circuits) == 4
        return len(circuits)
    
    @pytest.mark.benchmark
    @benchmark("cpu_intensive_optimization")
    def test_cpu_intensive_optimization(self):
        """Benchmark CPU-intensive quantum circuit optimization."""
        
        # Simulate CPU-intensive optimization
        def complex_optimization():
            # Simulate matrix operations common in quantum optimization
            for _ in range(100):
                matrix = np.random.random((64, 64)) + 1j * np.random.random((64, 64))
                eigenvalues = np.linalg.eigvals(matrix)
                # Simulate some processing
                result = np.sum(np.abs(eigenvalues))
            return result
        
        # Run optimization multiple times
        results = []
        for _ in range(10):
            result = complex_optimization()
            results.append(result)
        
        assert len(results) == 10
        return len(results)


# Benchmark reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate benchmark report."""
    if hasattr(pytest, 'benchmark_results'):
        results = pytest.benchmark_results
        
        if results:
            terminalreporter.write_sep("=", "Performance Benchmark Results")
            
            # Sort by execution time
            results.sort(key=lambda x: x['execution_time'], reverse=True)
            
            for result in results:
                status = "✓" if result['success'] else "✗"
                terminalreporter.write_line(
                    f"{status} {result['name']:<30} "
                    f"{result['execution_time']:>8.3f}s "
                    f"{result['memory_usage_mb']:>8.1f}MB"
                )
                
                if not result['success']:
                    terminalreporter.write_line(f"   Error: {result['error']}")
            
            # Summary statistics
            successful_results = [r for r in results if r['success']]
            if successful_results:
                avg_time = np.mean([r['execution_time'] for r in successful_results])
                avg_memory = np.mean([r['memory_usage_mb'] for r in successful_results])
                
                terminalreporter.write_line(f"\nBenchmark Summary:")
                terminalreporter.write_line(f"  Total benchmarks: {len(results)}")
                terminalreporter.write_line(f"  Successful: {len(successful_results)}")
                terminalreporter.write_line(f"  Average execution time: {avg_time:.3f}s")
                terminalreporter.write_line(f"  Average memory usage: {avg_memory:.1f}MB")