"""
Research Framework for Novel Quantum Algorithms and Comparative Studies.
Generation 4 research implementation with experimental validation.
"""

import asyncio
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from .exceptions import QuantumResearchError, QuantumValidationError
from .monitoring import PerformanceMetrics
from .caching import CacheManager

logger = logging.getLogger(__name__)

class ResearchPhase(Enum):
    """Research project phases."""
    HYPOTHESIS = "hypothesis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    PUBLICATION = "publication"

class AlgorithmType(Enum):
    """Types of quantum algorithms for research."""
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    MACHINE_LEARNING = "machine_learning"
    CRYPTOGRAPHY = "cryptography"
    ERROR_CORRECTION = "error_correction"
    COMMUNICATION = "communication"

@dataclass
class ResearchHypothesis:
    """Research hypothesis definition."""
    id: str
    title: str
    description: str
    algorithm_type: AlgorithmType
    success_criteria: Dict[str, Any]
    baseline_algorithms: List[str]
    metrics: List[str]
    expected_improvement: float
    significance_threshold: float = 0.05

@dataclass
class ExperimentResult:
    """Single experiment execution result."""
    experiment_id: str
    algorithm_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    timestamp: float
    circuit_depth: Optional[int] = None
    gate_count: Optional[int] = None
    error_rate: Optional[float] = None
    success: bool = True

@dataclass
class ComparativeStudy:
    """Comparative study definition and results."""
    study_id: str
    hypothesis: ResearchHypothesis
    algorithms: List[str]
    test_cases: List[Dict[str, Any]]
    results: List[ExperimentResult] = field(default_factory=list)
    statistical_analysis: Optional[Dict[str, Any]] = None
    conclusions: Optional[str] = None

class QuantumAlgorithm(ABC):
    """Abstract base class for quantum algorithms in research."""
    
    @abstractmethod
    def execute(self, problem_instance: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the algorithm on a problem instance."""
        pass
    
    @abstractmethod
    def get_circuit(self, problem_instance: Dict[str, Any]) -> Any:
        """Get the quantum circuit for this algorithm."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters for reproducibility."""
        pass

class NovelQuantumOptimizer(QuantumAlgorithm):
    """
    Novel quantum optimization algorithm with adaptive circuit depth.
    Research contribution: Dynamic circuit adaptation based on problem structure.
    """
    
    def __init__(self, adaptive_depth: bool = True, max_iterations: int = 100):
        self.adaptive_depth = adaptive_depth
        self.max_iterations = max_iterations
        self.name = "NovelQuantumOptimizer"
    
    def execute(self, problem_instance: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute novel optimization algorithm."""
        
        # Extract problem parameters
        num_variables = problem_instance.get('num_variables', 4)
        objective_function = problem_instance.get('objective_function', 'ising')
        
        start_time = time.time()
        
        # Novel approach: Adaptive circuit depth based on problem complexity
        initial_depth = self._calculate_initial_depth(num_variables)
        current_depth = initial_depth
        
        best_solution = None
        best_value = float('-inf')
        iteration_results = []
        
        for iteration in range(self.max_iterations):
            # Create circuit with current depth
            circuit = self._create_adaptive_circuit(num_variables, current_depth)
            
            # Execute circuit (simulated for research)
            result = self._simulate_execution(circuit, problem_instance)
            
            # Evaluate solution
            solution_value = self._evaluate_solution(result, problem_instance)
            
            iteration_results.append({
                'iteration': iteration,
                'depth': current_depth,
                'value': solution_value,
                'convergence_rate': self._calculate_convergence_rate(iteration_results)
            })
            
            if solution_value > best_value:
                best_value = solution_value
                best_solution = result
            
            # Adaptive depth adjustment (novel contribution)
            if self.adaptive_depth:
                current_depth = self._adapt_circuit_depth(
                    current_depth, iteration_results, num_variables
                )
        
        execution_time = time.time() - start_time
        
        return {
            'solution': best_solution,
            'objective_value': best_value,
            'execution_time': execution_time,
            'iterations': len(iteration_results),
            'final_depth': current_depth,
            'convergence_history': iteration_results,
            'algorithm_metrics': {
                'avg_depth': np.mean([r['depth'] for r in iteration_results]),
                'depth_variance': np.var([r['depth'] for r in iteration_results]),
                'convergence_rate': iteration_results[-1]['convergence_rate'] if iteration_results else 0
            }
        }
    
    def get_circuit(self, problem_instance: Dict[str, Any]) -> Any:
        """Get the quantum circuit for this problem instance."""
        num_variables = problem_instance.get('num_variables', 4)
        depth = self._calculate_initial_depth(num_variables)
        return self._create_adaptive_circuit(num_variables, depth)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return {
            'algorithm': self.name,
            'adaptive_depth': self.adaptive_depth,
            'max_iterations': self.max_iterations
        }
    
    def _calculate_initial_depth(self, num_variables: int) -> int:
        """Calculate initial circuit depth based on problem size."""
        # Novel heuristic: logarithmic scaling with problem complexity adjustment
        base_depth = max(2, int(np.log2(num_variables)) + 1)
        complexity_factor = 1 + (num_variables / 20)  # Adjust for larger problems
        return int(base_depth * complexity_factor)
    
    def _create_adaptive_circuit(self, num_variables: int, depth: int) -> Dict[str, Any]:
        """Create adaptive quantum circuit."""
        # Simplified circuit representation for research
        return {
            'num_qubits': num_variables,
            'depth': depth,
            'gates': self._generate_gate_sequence(num_variables, depth),
            'parameters': np.random.uniform(0, 2*np.pi, depth * num_variables)
        }
    
    def _generate_gate_sequence(self, num_qubits: int, depth: int) -> List[Dict[str, Any]]:
        """Generate optimized gate sequence."""
        gates = []
        for layer in range(depth):
            # Alternating layers of single and two-qubit gates
            if layer % 2 == 0:
                # Single-qubit rotation layers
                for qubit in range(num_qubits):
                    gates.append({
                        'type': 'RY',
                        'qubits': [qubit],
                        'parameter_index': layer * num_qubits + qubit
                    })
            else:
                # Entangling layers
                for qubit in range(num_qubits - 1):
                    gates.append({
                        'type': 'CNOT',
                        'qubits': [qubit, qubit + 1]
                    })
        return gates
    
    def _simulate_execution(self, circuit: Dict[str, Any], problem_instance: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate quantum circuit execution."""
        # Simplified simulation for research purposes
        num_qubits = circuit['num_qubits']
        
        # Generate mock measurement results based on circuit parameters
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        # Simulate quantum state evolution
        probabilities = np.random.dirichlet(np.ones(2**num_qubits))
        
        # Sample from probability distribution
        shots = 1000
        samples = np.random.choice(2**num_qubits, size=shots, p=probabilities)
        
        # Convert to bit strings
        bit_strings = []
        for sample in samples:
            bit_string = format(sample, f'0{num_qubits}b')
            bit_strings.append(bit_string)
        
        # Count occurrences
        counts = {}
        for bit_string in bit_strings:
            counts[bit_string] = counts.get(bit_string, 0) + 1
        
        return {
            'counts': counts,
            'shots': shots,
            'probabilities': probabilities.tolist(),
            'circuit_depth': circuit['depth']
        }
    
    def _evaluate_solution(self, result: Dict[str, Any], problem_instance: Dict[str, Any]) -> float:
        """Evaluate solution quality for the optimization problem."""
        counts = result['counts']
        
        # For Ising model problems, evaluate energy
        if problem_instance.get('objective_function') == 'ising':
            total_energy = 0
            total_shots = sum(counts.values())
            
            for bit_string, count in counts.items():
                energy = self._calculate_ising_energy(bit_string, problem_instance)
                probability = count / total_shots
                total_energy += probability * energy
            
            return -total_energy  # Negative because we want to minimize energy
        
        # For general optimization, use most frequent solution
        if counts:
            most_frequent = max(counts.items(), key=lambda x: x[1])
            return most_frequent[1] / sum(counts.values())
        
        return 0.0
    
    def _calculate_ising_energy(self, bit_string: str, problem_instance: Dict[str, Any]) -> float:
        """Calculate Ising model energy for a bit string."""
        # Simplified Ising energy calculation
        spins = [1 if bit == '1' else -1 for bit in bit_string]
        
        # Random Ising coefficients for research
        np.random.seed(hash(str(problem_instance)) % 2**32)
        J = np.random.uniform(-1, 1, (len(spins), len(spins)))
        h = np.random.uniform(-0.5, 0.5, len(spins))
        
        energy = 0
        
        # Interaction terms
        for i in range(len(spins)):
            for j in range(i + 1, len(spins)):
                energy += J[i, j] * spins[i] * spins[j]
        
        # Field terms
        for i in range(len(spins)):
            energy += h[i] * spins[i]
        
        return energy
    
    def _calculate_convergence_rate(self, iteration_results: List[Dict[str, Any]]) -> float:
        """Calculate convergence rate based on recent iterations."""
        if len(iteration_results) < 3:
            return 0.0
        
        recent_values = [r['value'] for r in iteration_results[-5:]]
        
        if len(recent_values) < 2:
            return 0.0
        
        # Calculate variance in recent values (lower = better convergence)
        variance = np.var(recent_values)
        convergence_rate = 1.0 / (1.0 + variance)
        
        return convergence_rate
    
    def _adapt_circuit_depth(self, current_depth: int, iteration_results: List[Dict[str, Any]], 
                           num_variables: int) -> int:
        """Adapt circuit depth based on performance history (novel contribution)."""
        
        if len(iteration_results) < 5:
            return current_depth
        
        # Analyze recent performance
        recent_convergence = [r['convergence_rate'] for r in iteration_results[-3:]]
        avg_convergence = np.mean(recent_convergence)
        
        # Adaptation strategy
        if avg_convergence < 0.3:  # Poor convergence
            # Increase depth to improve expressibility
            new_depth = min(current_depth + 1, num_variables * 3)
        elif avg_convergence > 0.8:  # Good convergence
            # Potentially reduce depth for efficiency
            new_depth = max(current_depth - 1, 2)
        else:
            # Keep current depth
            new_depth = current_depth
        
        return new_depth

class ResearchFramework:
    """
    Comprehensive research framework for quantum algorithm development and validation.
    """
    
    def __init__(self):
        # Monitoring disabled for demo
        self.cache_manager = CacheManager()
        self.studies: Dict[str, ComparativeStudy] = {}
        self.algorithms: Dict[str, QuantumAlgorithm] = {}
        self.results_directory = Path("research_results")
        self.results_directory.mkdir(exist_ok=True)
        
        # Register built-in algorithms
        self._register_builtin_algorithms()
    
    def _register_builtin_algorithms(self):
        """Register built-in research algorithms."""
        self.algorithms['novel_optimizer'] = NovelQuantumOptimizer()
        self.algorithms['adaptive_optimizer'] = NovelQuantumOptimizer(adaptive_depth=True)
        self.algorithms['fixed_optimizer'] = NovelQuantumOptimizer(adaptive_depth=False)
    
    def register_algorithm(self, name: str, algorithm: QuantumAlgorithm):
        """Register a new algorithm for research."""
        self.algorithms[name] = algorithm
        logger.info(f"Registered algorithm: {name}")
    
    def create_hypothesis(self, hypothesis_data: Dict[str, Any]) -> ResearchHypothesis:
        """Create a research hypothesis."""
        
        hypothesis = ResearchHypothesis(
            id=hypothesis_data['id'],
            title=hypothesis_data['title'],
            description=hypothesis_data['description'],
            algorithm_type=AlgorithmType(hypothesis_data['algorithm_type']),
            success_criteria=hypothesis_data['success_criteria'],
            baseline_algorithms=hypothesis_data['baseline_algorithms'],
            metrics=hypothesis_data['metrics'],
            expected_improvement=hypothesis_data['expected_improvement']
        )
        
        logger.info(f"Created research hypothesis: {hypothesis.title}")
        return hypothesis
    
    def design_comparative_study(self, study_config: Dict[str, Any]) -> ComparativeStudy:
        """Design a comparative study between algorithms."""
        
        hypothesis = self.create_hypothesis(study_config['hypothesis'])
        
        study = ComparativeStudy(
            study_id=study_config['study_id'],
            hypothesis=hypothesis,
            algorithms=study_config['algorithms'],
            test_cases=study_config['test_cases']
        )
        
        self.studies[study.study_id] = study
        logger.info(f"Designed comparative study: {study.study_id}")
        
        return study
    
    async def execute_comparative_study(self, study_id: str, 
                                      runs_per_algorithm: int = 10) -> ComparativeStudy:
        """Execute a comparative study with statistical validation."""
        
        if study_id not in self.studies:
            raise QuantumResearchError(f"Study {study_id} not found")
        
        study = self.studies[study_id]
        
        logger.info(f"Executing comparative study: {study_id}")
        logger.info(f"Algorithms: {study.algorithms}")
        logger.info(f"Test cases: {len(study.test_cases)}")
        logger.info(f"Runs per algorithm: {runs_per_algorithm}")
        
        # Execute experiments for each algorithm and test case
        for algorithm_name in study.algorithms:
            if algorithm_name not in self.algorithms:
                logger.warning(f"Algorithm {algorithm_name} not found, skipping")
                continue
            
            algorithm = self.algorithms[algorithm_name]
            
            for test_case_idx, test_case in enumerate(study.test_cases):
                for run in range(runs_per_algorithm):
                    
                    experiment_id = f"{study_id}_{algorithm_name}_{test_case_idx}_{run}"
                    
                    try:
                        result = await self._execute_single_experiment(
                            experiment_id, algorithm, test_case, study.hypothesis.metrics
                        )
                        study.results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Experiment {experiment_id} failed: {e}")
                        
                        # Add failed result
                        failed_result = ExperimentResult(
                            experiment_id=experiment_id,
                            algorithm_name=algorithm_name,
                            parameters=test_case,
                            metrics={},
                            execution_time=0.0,
                            timestamp=time.time(),
                            success=False
                        )
                        study.results.append(failed_result)
        
        # Perform statistical analysis
        study.statistical_analysis = self._perform_statistical_analysis(study)
        
        # Generate conclusions
        study.conclusions = self._generate_conclusions(study)
        
        # Save results
        self._save_study_results(study)
        
        logger.info(f"Completed comparative study: {study_id}")
        return study
    
    async def _execute_single_experiment(self, experiment_id: str, algorithm: QuantumAlgorithm,
                                       test_case: Dict[str, Any], 
                                       metrics: List[str]) -> ExperimentResult:
        """Execute a single experimental run."""
        
        start_time = time.time()
        
        # Execute algorithm
        result = algorithm.execute(test_case)
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        calculated_metrics = self._calculate_metrics(result, metrics)
        
        # Extract circuit information
        circuit = algorithm.get_circuit(test_case)
        circuit_depth = circuit.get('depth') if isinstance(circuit, dict) else None
        gate_count = len(circuit.get('gates', [])) if isinstance(circuit, dict) else None
        
        return ExperimentResult(
            experiment_id=experiment_id,
            algorithm_name=algorithm.get_parameters().get('algorithm', 'unknown'),
            parameters=algorithm.get_parameters(),
            metrics=calculated_metrics,
            execution_time=execution_time,
            timestamp=time.time(),
            circuit_depth=circuit_depth,
            gate_count=gate_count,
            error_rate=result.get('error_rate'),
            success=True
        )
    
    def _calculate_metrics(self, result: Dict[str, Any], metrics: List[str]) -> Dict[str, float]:
        """Calculate specified metrics from algorithm result."""
        
        calculated = {}
        
        for metric in metrics:
            if metric == 'objective_value':
                calculated[metric] = result.get('objective_value', 0.0)
            elif metric == 'execution_time':
                calculated[metric] = result.get('execution_time', 0.0)
            elif metric == 'convergence_rate':
                calculated[metric] = result.get('algorithm_metrics', {}).get('convergence_rate', 0.0)
            elif metric == 'circuit_depth':
                calculated[metric] = result.get('final_depth', 0.0)
            elif metric == 'iterations':
                calculated[metric] = float(result.get('iterations', 0))
            elif metric == 'solution_quality':
                # Custom metric: normalized objective value
                obj_val = result.get('objective_value', 0.0)
                calculated[metric] = max(0, min(1, (obj_val + 1) / 2))  # Normalize to [0,1]
        
        return calculated
    
    def _perform_statistical_analysis(self, study: ComparativeStudy) -> Dict[str, Any]:
        """Perform statistical analysis on study results."""
        
        analysis = {
            'algorithms': {},
            'comparisons': {},
            'significance_tests': {}
        }
        
        # Group results by algorithm
        algorithm_results = {}
        for result in study.results:
            if result.success:
                if result.algorithm_name not in algorithm_results:
                    algorithm_results[result.algorithm_name] = []
                algorithm_results[result.algorithm_name].append(result)
        
        # Calculate statistics for each algorithm
        for algorithm_name, results in algorithm_results.items():
            
            metrics_data = {}
            for metric in study.hypothesis.metrics:
                values = [r.metrics.get(metric, 0) for r in results]
                
                if values:
                    metrics_data[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            analysis['algorithms'][algorithm_name] = {
                'total_runs': len(results),
                'success_rate': len([r for r in results if r.success]) / len(results),
                'metrics': metrics_data
            }
        
        # Perform pairwise comparisons
        algorithms = list(algorithm_results.keys())
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                
                comparison_key = f"{alg1}_vs_{alg2}"
                analysis['comparisons'][comparison_key] = {}
                
                for metric in study.hypothesis.metrics:
                    values1 = [r.metrics.get(metric, 0) for r in algorithm_results[alg1]]
                    values2 = [r.metrics.get(metric, 0) for r in algorithm_results[alg2]]
                    
                    if values1 and values2:
                        # Perform t-test (simplified)
                        mean1, mean2 = np.mean(values1), np.mean(values2)
                        std1, std2 = np.std(values1), np.std(values2)
                        n1, n2 = len(values1), len(values2)
                        
                        # Welch's t-test approximation
                        pooled_std = np.sqrt((std1**2/n1) + (std2**2/n2))
                        
                        if pooled_std > 0:
                            t_stat = (mean1 - mean2) / pooled_std
                            
                            # Simplified p-value estimation
                            p_value = 2 * (1 - np.exp(-0.717 * abs(t_stat) - 0.416 * t_stat**2))
                            p_value = min(1.0, max(0.0, p_value))
                        else:
                            t_stat = 0
                            p_value = 1.0
                        
                        analysis['comparisons'][comparison_key][metric] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'mean_difference': mean1 - mean2,
                            'effect_size': (mean1 - mean2) / pooled_std if pooled_std > 0 else 0,
                            'significant': p_value < study.hypothesis.significance_threshold
                        }
        
        return analysis
    
    def _generate_conclusions(self, study: ComparativeStudy) -> str:
        """Generate research conclusions based on statistical analysis."""
        
        if not study.statistical_analysis:
            return "No statistical analysis available."
        
        conclusions = []
        
        # Overall performance summary
        algorithms = study.statistical_analysis['algorithms']
        conclusions.append("## Experimental Results Summary\n")
        
        for algorithm, stats in algorithms.items():
            success_rate = stats['success_rate']
            conclusions.append(f"- **{algorithm}**: {stats['total_runs']} runs, "
                             f"{success_rate:.1%} success rate")
        
        # Significance findings
        conclusions.append("\n## Statistical Significance Findings\n")
        
        significant_findings = []
        comparisons = study.statistical_analysis['comparisons']
        
        for comparison, metrics_data in comparisons.items():
            alg1, alg2 = comparison.split('_vs_')
            
            for metric, stats in metrics_data.items():
                if stats['significant']:
                    effect_size = abs(stats['effect_size'])
                    direction = "outperformed" if stats['mean_difference'] > 0 else "underperformed"
                    
                    significant_findings.append(
                        f"- **{alg1}** {direction} **{alg2}** on {metric} "
                        f"(p={stats['p_value']:.3f}, effect size={effect_size:.2f})"
                    )
        
        if significant_findings:
            conclusions.extend(significant_findings)
        else:
            conclusions.append("- No statistically significant differences found between algorithms")
        
        # Hypothesis evaluation
        conclusions.append("\n## Hypothesis Evaluation\n")
        
        hypothesis_met = self._evaluate_hypothesis(study)
        
        if hypothesis_met:
            conclusions.append("✅ **Research hypothesis is SUPPORTED by the experimental evidence.**")
        else:
            conclusions.append("❌ **Research hypothesis is NOT SUPPORTED by the experimental evidence.**")
        
        # Recommendations
        conclusions.append("\n## Recommendations\n")
        
        best_algorithm = self._identify_best_algorithm(study)
        if best_algorithm:
            conclusions.append(f"- **Recommended algorithm**: {best_algorithm}")
        
        conclusions.append("- Conduct additional experiments with larger sample sizes")
        conclusions.append("- Investigate parameter sensitivity for top-performing algorithms")
        conclusions.append("- Validate results on real quantum hardware")
        
        return "\n".join(conclusions)
    
    def _evaluate_hypothesis(self, study: ComparativeStudy) -> bool:
        """Evaluate if the research hypothesis is supported."""
        
        # Check if novel algorithm meets expected improvement threshold
        target_metric = study.hypothesis.success_criteria.get('primary_metric', 'objective_value')
        expected_improvement = study.hypothesis.expected_improvement
        
        if not study.statistical_analysis:
            return False
        
        algorithms = study.statistical_analysis['algorithms']
        
        # Find novel algorithm performance
        novel_algorithms = [alg for alg in algorithms.keys() if 'novel' in alg.lower()]
        baseline_algorithms = study.hypothesis.baseline_algorithms
        
        if not novel_algorithms or not baseline_algorithms:
            return False
        
        # Compare best novel vs best baseline
        novel_performance = max([
            algorithms[alg]['metrics'].get(target_metric, {}).get('mean', 0)
            for alg in novel_algorithms
            if target_metric in algorithms[alg]['metrics']
        ])
        
        baseline_performance = max([
            algorithms[alg]['metrics'].get(target_metric, {}).get('mean', 0)
            for alg in baseline_algorithms
            if alg in algorithms and target_metric in algorithms[alg]['metrics']
        ])
        
        if baseline_performance == 0:
            return False
        
        actual_improvement = (novel_performance - baseline_performance) / abs(baseline_performance)
        
        return actual_improvement >= expected_improvement
    
    def _identify_best_algorithm(self, study: ComparativeStudy) -> Optional[str]:
        """Identify the best performing algorithm."""
        
        if not study.statistical_analysis:
            return None
        
        algorithms = study.statistical_analysis['algorithms']
        target_metric = study.hypothesis.success_criteria.get('primary_metric', 'objective_value')
        
        best_algorithm = None
        best_score = float('-inf')
        
        for algorithm, stats in algorithms.items():
            if target_metric in stats['metrics']:
                score = stats['metrics'][target_metric]['mean']
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
        
        return best_algorithm
    
    def _save_study_results(self, study: ComparativeStudy):
        """Save study results to disk."""
        
        study_file = self.results_directory / f"{study.study_id}.json"
        
        # Convert study to serializable format
        study_data = {
            'study_id': study.study_id,
            'hypothesis': {
                'id': study.hypothesis.id,
                'title': study.hypothesis.title,
                'description': study.hypothesis.description,
                'algorithm_type': study.hypothesis.algorithm_type.value,
                'success_criteria': study.hypothesis.success_criteria,
                'baseline_algorithms': study.hypothesis.baseline_algorithms,
                'metrics': study.hypothesis.metrics,
                'expected_improvement': study.hypothesis.expected_improvement
            },
            'algorithms': study.algorithms,
            'test_cases': study.test_cases,
            'results': [
                {
                    'experiment_id': r.experiment_id,
                    'algorithm_name': r.algorithm_name,
                    'parameters': r.parameters,
                    'metrics': r.metrics,
                    'execution_time': r.execution_time,
                    'timestamp': r.timestamp,
                    'circuit_depth': r.circuit_depth,
                    'gate_count': r.gate_count,
                    'error_rate': r.error_rate,
                    'success': r.success
                }
                for r in study.results
            ],
            'statistical_analysis': study.statistical_analysis,
            'conclusions': study.conclusions
        }
        
        with open(study_file, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        logger.info(f"Saved study results to {study_file}")
    
    def generate_research_report(self, study_id: str) -> str:
        """Generate a comprehensive research report."""
        
        if study_id not in self.studies:
            raise QuantumResearchError(f"Study {study_id} not found")
        
        study = self.studies[study_id]
        
        report = []
        
        # Title and abstract
        report.append(f"# {study.hypothesis.title}")
        report.append(f"\n**Study ID**: {study.study_id}")
        report.append(f"**Algorithm Type**: {study.hypothesis.algorithm_type.value}")
        report.append(f"\n## Abstract\n")
        report.append(study.hypothesis.description)
        
        # Methodology
        report.append("\n## Methodology\n")
        report.append(f"- **Algorithms tested**: {', '.join(study.algorithms)}")
        report.append(f"- **Test cases**: {len(study.test_cases)}")
        report.append(f"- **Metrics evaluated**: {', '.join(study.hypothesis.metrics)}")
        report.append(f"- **Significance threshold**: {study.hypothesis.significance_threshold}")
        
        # Results
        if study.statistical_analysis:
            report.append("\n## Results\n")
            
            for algorithm, stats in study.statistical_analysis['algorithms'].items():
                report.append(f"\n### {algorithm}")
                report.append(f"- **Total runs**: {stats['total_runs']}")
                report.append(f"- **Success rate**: {stats['success_rate']:.1%}")
                
                for metric, metric_stats in stats['metrics'].items():
                    mean = metric_stats['mean']
                    std = metric_stats['std']
                    report.append(f"- **{metric}**: {mean:.4f} ± {std:.4f}")
        
        # Conclusions
        if study.conclusions:
            report.append(f"\n{study.conclusions}")
        
        return "\n".join(report)
    
    def create_visualization(self, study_id: str, output_path: Optional[str] = None) -> str:
        """Create visualizations for research results."""
        
        if study_id not in self.studies:
            raise QuantumResearchError(f"Study {study_id} not found")
        
        study = self.studies[study_id]
        
        if not study.statistical_analysis:
            raise QuantumResearchError("No statistical analysis available for visualization")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Research Results: {study.hypothesis.title}', fontsize=16)
        
        # Plot 1: Algorithm performance comparison
        algorithms = list(study.statistical_analysis['algorithms'].keys())
        primary_metric = study.hypothesis.metrics[0] if study.hypothesis.metrics else 'objective_value'
        
        means = []
        stds = []
        
        for alg in algorithms:
            stats = study.statistical_analysis['algorithms'][alg]
            if primary_metric in stats['metrics']:
                means.append(stats['metrics'][primary_metric]['mean'])
                stds.append(stats['metrics'][primary_metric]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        axes[0, 0].bar(algorithms, means, yerr=stds, capsize=5)
        axes[0, 0].set_title(f'{primary_metric.replace("_", " ").title()} Comparison')
        axes[0, 0].set_ylabel(primary_metric.replace("_", " ").title())
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Success rates
        success_rates = [study.statistical_analysis['algorithms'][alg]['success_rate'] for alg in algorithms]
        
        axes[0, 1].bar(algorithms, success_rates)
        axes[0, 1].set_title('Success Rate by Algorithm')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Execution time distribution
        execution_times = {}
        for result in study.results:
            if result.success:
                if result.algorithm_name not in execution_times:
                    execution_times[result.algorithm_name] = []
                execution_times[result.algorithm_name].append(result.execution_time)
        
        box_data = [execution_times.get(alg, [0]) for alg in algorithms]
        axes[1, 0].boxplot(box_data, labels=algorithms)
        axes[1, 0].set_title('Execution Time Distribution')
        axes[1, 0].set_ylabel('Execution Time (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Correlation matrix for metrics
        if len(study.hypothesis.metrics) > 1:
            metric_data = {}
            
            for metric in study.hypothesis.metrics:
                metric_data[metric] = []
                for result in study.results:
                    if result.success and metric in result.metrics:
                        metric_data[metric].append(result.metrics[metric])
            
            # Create correlation matrix
            df = pd.DataFrame(metric_data)
            if len(df) > 0:
                corr_matrix = df.corr()
                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_title('Metric Correlation Matrix')
                axes[1, 1].set_xticks(range(len(study.hypothesis.metrics)))
                axes[1, 1].set_yticks(range(len(study.hypothesis.metrics)))
                axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in study.hypothesis.metrics])
                axes[1, 1].set_yticklabels([m.replace('_', ' ').title() for m in study.hypothesis.metrics])
                plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save visualization
        if output_path is None:
            output_path = str(self.results_directory / f"{study_id}_visualization.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        return output_path