#!/usr/bin/env python3
"""
Quantum DevOps CI Research Framework Demonstration.
Showcases novel algorithm development and comparative studies.
"""

import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import research framework
from src.quantum_devops_ci.research_framework import (
    ResearchFramework, 
    AlgorithmType,
    NovelQuantumOptimizer
)

async def main():
    """Demonstrate the research framework with a comparative study."""
    
    print("🧪 Quantum DevOps CI - Research Framework Demo")
    print("=" * 60)
    
    # Initialize research framework
    framework = ResearchFramework()
    
    # Define research hypothesis
    hypothesis_data = {
        'id': 'adaptive_depth_hypothesis',
        'title': 'Adaptive Circuit Depth Improves Quantum Optimization Performance',
        'description': (
            'We hypothesize that dynamically adapting quantum circuit depth '
            'based on convergence patterns leads to better optimization results '
            'compared to fixed-depth approaches, with at least 15% improvement '
            'in solution quality.'
        ),
        'algorithm_type': 'optimization',
        'success_criteria': {
            'primary_metric': 'objective_value',
            'minimum_improvement': 0.15
        },
        'baseline_algorithms': ['fixed_optimizer'],
        'metrics': ['objective_value', 'execution_time', 'convergence_rate', 'circuit_depth'],
        'expected_improvement': 0.15
    }
    
    # Define test cases for comparative study
    test_cases = [
        {
            'num_variables': 4,
            'objective_function': 'ising',
            'problem_complexity': 'low',
            'problem_id': 'ising_4_low'
        },
        {
            'num_variables': 6,
            'objective_function': 'ising',
            'problem_complexity': 'medium',
            'problem_id': 'ising_6_medium'
        },
        {
            'num_variables': 8,
            'objective_function': 'ising',
            'problem_complexity': 'high',
            'problem_id': 'ising_8_high'
        }
    ]
    
    # Design comparative study
    study_config = {
        'study_id': 'adaptive_vs_fixed_depth_2025',
        'hypothesis': hypothesis_data,
        'algorithms': ['novel_optimizer', 'adaptive_optimizer', 'fixed_optimizer'],
        'test_cases': test_cases
    }
    
    print("\n📋 Study Configuration:")
    print(f"- Study ID: {study_config['study_id']}")
    print(f"- Algorithms: {study_config['algorithms']}")
    print(f"- Test Cases: {len(study_config['test_cases'])}")
    print(f"- Expected Improvement: {hypothesis_data['expected_improvement']:.1%}")
    
    # Create and execute comparative study
    try:
        print("\n🔬 Designing comparative study...")
        study = framework.design_comparative_study(study_config)
        
        print("✅ Study designed successfully")
        
        print("\n🚀 Executing comparative study...")
        print("This may take a few minutes as we run multiple experiments...")
        
        # Execute with multiple runs for statistical significance
        completed_study = await framework.execute_comparative_study(
            study.study_id, 
            runs_per_algorithm=5  # Reduced for demo
        )
        
        print("✅ Study execution completed")
        
        # Display results summary
        print("\n📊 Results Summary:")
        print("-" * 40)
        
        if completed_study.statistical_analysis:
            algorithms = completed_study.statistical_analysis['algorithms']
            
            for algorithm, stats in algorithms.items():
                print(f"\n🔹 {algorithm}:")
                print(f"   Runs: {stats['total_runs']}")
                print(f"   Success Rate: {stats['success_rate']:.1%}")
                
                for metric, metric_stats in stats['metrics'].items():
                    mean = metric_stats['mean']
                    std = metric_stats['std']
                    print(f"   {metric}: {mean:.4f} ± {std:.4f}")
        
        # Display statistical significance
        print("\n🔍 Statistical Analysis:")
        print("-" * 40)
        
        if completed_study.statistical_analysis and 'comparisons' in completed_study.statistical_analysis:
            comparisons = completed_study.statistical_analysis['comparisons']
            
            for comparison, metrics_data in comparisons.items():
                alg1, alg2 = comparison.split('_vs_')
                print(f"\n{alg1} vs {alg2}:")
                
                for metric, stats in metrics_data.items():
                    significance = "✅ SIGNIFICANT" if stats['significant'] else "❌ Not significant"
                    effect_size = abs(stats['effect_size'])
                    
                    print(f"   {metric}: p={stats['p_value']:.3f}, "
                          f"effect={effect_size:.2f}, {significance}")
        
        # Display conclusions
        print("\n📝 Research Conclusions:")
        print("-" * 40)
        
        if completed_study.conclusions:
            print(completed_study.conclusions)
        
        # Generate full research report
        print("\n📄 Generating research report...")
        
        report = framework.generate_research_report(completed_study.study_id)
        report_file = Path("research_results") / f"{completed_study.study_id}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Full report saved to: {report_file}")
        
        # Create visualization
        try:
            print("\n📈 Creating visualizations...")
            viz_path = framework.create_visualization(completed_study.study_id)
            print(f"✅ Visualization saved to: {viz_path}")
        except Exception as e:
            print(f"⚠️  Visualization failed (matplotlib may not be available): {e}")
        
        # Summary statistics
        total_experiments = len(completed_study.results)
        successful_experiments = len([r for r in completed_study.results if r.success])
        
        print("\n📈 Experiment Summary:")
        print("-" * 40)
        print(f"Total Experiments: {total_experiments}")
        print(f"Successful: {successful_experiments}")
        print(f"Success Rate: {successful_experiments/total_experiments:.1%}")
        
        # Performance insights
        if completed_study.statistical_analysis:
            best_algorithm = framework._identify_best_algorithm(completed_study)
            if best_algorithm:
                print(f"Best Performing Algorithm: {best_algorithm}")
        
        # Hypothesis evaluation
        hypothesis_supported = framework._evaluate_hypothesis(completed_study)
        result = "SUPPORTED ✅" if hypothesis_supported else "NOT SUPPORTED ❌"
        print(f"Research Hypothesis: {result}")
        
        print("\n🎉 Research demonstration completed successfully!")
        print("\nKey Research Contributions:")
        print("- Novel adaptive circuit depth algorithm implemented")
        print("- Comparative study with statistical validation")
        print("- Reproducible experimental framework")
        print("- Automated research report generation")
        print("\nThis framework enables rapid prototyping and validation")
        print("of quantum algorithms with rigorous scientific methodology.")
        
    except Exception as e:
        logger.error(f"Research demonstration failed: {e}")
        print(f"\n❌ Error during research demonstration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())