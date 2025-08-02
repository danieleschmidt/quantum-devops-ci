#!/usr/bin/env python3
"""
Automated metrics collection script for quantum-devops-ci project.

This script collects various metrics from different sources and aggregates them
for monitoring and reporting purposes.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from github import Github

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metrics-collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates project metrics from various sources."""
    
    def __init__(self, config_path: str = '.github/project-metrics.json'):
        """Initialize the metrics collector.
        
        Args:
            config_path: Path to the project metrics configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.github_client = self._init_github_client()
        self.metrics = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load project metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _init_github_client(self) -> Optional[Github]:
        """Initialize GitHub API client."""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            logger.warning("GITHUB_TOKEN not found. GitHub metrics will be limited.")
            return None
        return Github(token)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        logger.info("Starting metrics collection...")
        
        # Collect different categories of metrics
        self.metrics.update({
            'timestamp': datetime.utcnow().isoformat(),
            'repository': self._collect_repository_metrics(),
            'development': self._collect_development_metrics(),
            'performance': self._collect_performance_metrics(),
            'security': self._collect_security_metrics(),
            'operational': self._collect_operational_metrics(),
            'quantum': self._collect_quantum_metrics(),
            'cost': self._collect_cost_metrics()
        })
        
        logger.info("Metrics collection completed")
        return self.metrics
    
    def _collect_repository_metrics(self) -> Dict[str, Any]:
        """Collect repository-related metrics."""
        logger.info("Collecting repository metrics...")
        metrics = {}
        
        if not self.github_client:
            return metrics
        
        try:
            repo_name = self.config.get('repository', {}).get('fullName')
            if not repo_name:
                logger.warning("Repository name not found in config")
                return metrics
            
            repo = self.github_client.get_repo(repo_name)
            
            # Basic repository stats
            metrics['stars'] = repo.stargazers_count
            metrics['forks'] = repo.forks_count
            metrics['watchers'] = repo.watchers_count
            metrics['open_issues'] = repo.open_issues_count
            metrics['subscribers'] = repo.subscribers_count
            
            # Recent activity (last 30 days)
            since = datetime.utcnow() - timedelta(days=30)
            
            # Commits
            commits = list(repo.get_commits(since=since))
            metrics['commits_last_30_days'] = len(commits)
            
            # Pull requests
            prs = list(repo.get_pulls(state='all', sort='created', direction='desc'))
            recent_prs = [pr for pr in prs if pr.created_at >= since]
            metrics['prs_last_30_days'] = len(recent_prs)
            
            # Issues
            issues = list(repo.get_issues(state='all', since=since))
            # Filter out PRs (GitHub treats PRs as issues)
            issues = [issue for issue in issues if not issue.pull_request]
            metrics['issues_last_30_days'] = len(issues)
            
            # Contributors
            contributors = list(repo.get_contributors())
            metrics['total_contributors'] = len(contributors)
            
        except Exception as e:
            logger.error(f"Error collecting repository metrics: {e}")
        
        return metrics
    
    def _collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development-related metrics."""
        logger.info("Collecting development metrics...")
        metrics = {}
        
        # Code quality metrics from local files
        try:
            # Test coverage (if coverage report exists)
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    metrics['test_coverage'] = coverage_data.get('totals', {}).get('percent_covered', 0)
            
            # ESLint results
            eslint_file = Path('eslint-results.json')
            if eslint_file.exists():
                with open(eslint_file, 'r') as f:
                    eslint_data = json.load(f)
                    metrics['eslint_errors'] = sum(result['errorCount'] for result in eslint_data)
                    metrics['eslint_warnings'] = sum(result['warningCount'] for result in eslint_data)
            
            # Python linting results
            flake8_file = Path('flake8-results.txt')
            if flake8_file.exists():
                with open(flake8_file, 'r') as f:
                    lines = f.readlines()
                    metrics['flake8_issues'] = len([line for line in lines if line.strip()])
            
            # Count lines of code
            metrics['lines_of_code'] = self._count_lines_of_code()
            
        except Exception as e:
            logger.error(f"Error collecting development metrics: {e}")
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        logger.info("Collecting performance metrics...")
        metrics = {}
        
        try:
            # Build time metrics (from CI logs if available)
            ci_log_file = Path('ci-build-times.json')
            if ci_log_file.exists():
                with open(ci_log_file, 'r') as f:
                    build_data = json.load(f)
                    recent_builds = build_data.get('recent_builds', [])
                    if recent_builds:
                        avg_build_time = sum(build['duration'] for build in recent_builds) / len(recent_builds)
                        metrics['average_build_time_seconds'] = avg_build_time
            
            # Test execution times
            test_results_file = Path('test-results.json')
            if test_results_file.exists():
                with open(test_results_file, 'r') as f:
                    test_data = json.load(f)
                    metrics['test_execution_time_seconds'] = test_data.get('duration', 0)
                    metrics['test_success_rate'] = test_data.get('success_rate', 0)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        logger.info("Collecting security metrics...")
        metrics = {}
        
        try:
            # Security scan results
            security_scan_file = Path('security-scan-results.json')
            if security_scan_file.exists():
                with open(security_scan_file, 'r') as f:
                    security_data = json.load(f)
                    metrics['critical_vulnerabilities'] = security_data.get('critical', 0)
                    metrics['high_vulnerabilities'] = security_data.get('high', 0)
                    metrics['medium_vulnerabilities'] = security_data.get('medium', 0)
                    metrics['low_vulnerabilities'] = security_data.get('low', 0)
            
            # Dependency vulnerabilities
            safety_results_file = Path('safety-results.json')
            if safety_results_file.exists():
                with open(safety_results_file, 'r') as f:
                    safety_data = json.load(f)
                    metrics['dependency_vulnerabilities'] = len(safety_data.get('vulnerabilities', []))
            
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
        
        return metrics
    
    def _collect_operational_metrics(self) -> Dict[str, Any]:
        """Collect operational metrics."""
        logger.info("Collecting operational metrics...")
        metrics = {}
        
        try:
            # Deployment metrics (if deployment logs exist)
            deployment_file = Path('deployment-metrics.json')
            if deployment_file.exists():
                with open(deployment_file, 'r') as f:
                    deployment_data = json.load(f)
                    metrics['deployment_frequency'] = deployment_data.get('frequency', 0)
                    metrics['deployment_success_rate'] = deployment_data.get('success_rate', 0)
                    metrics['rollback_rate'] = deployment_data.get('rollback_rate', 0)
            
            # Service health metrics
            health_file = Path('health-metrics.json')
            if health_file.exists():
                with open(health_file, 'r') as f:
                    health_data = json.load(f)
                    metrics['uptime_percentage'] = health_data.get('uptime', 0)
                    metrics['response_time_ms'] = health_data.get('response_time', 0)
            
        except Exception as e:
            logger.error(f"Error collecting operational metrics: {e}")
        
        return metrics
    
    def _collect_quantum_metrics(self) -> Dict[str, Any]:
        """Collect quantum computing specific metrics."""
        logger.info("Collecting quantum metrics...")
        metrics = {}
        
        try:
            # Quantum test results
            quantum_results_file = Path('quantum-test-results.json')
            if quantum_results_file.exists():
                with open(quantum_results_file, 'r') as f:
                    quantum_data = json.load(f)
                    metrics['circuits_tested'] = quantum_data.get('circuits_tested', 0)
                    metrics['average_fidelity'] = quantum_data.get('average_fidelity', 0)
                    metrics['gate_count_reduction'] = quantum_data.get('gate_count_reduction', 0)
                    metrics['depth_reduction'] = quantum_data.get('depth_reduction', 0)
            
            # Hardware usage metrics
            hardware_usage_file = Path('quantum-hardware-usage.json')
            if hardware_usage_file.exists():
                with open(hardware_usage_file, 'r') as f:
                    hardware_data = json.load(f)
                    metrics['queue_time_seconds'] = hardware_data.get('average_queue_time', 0)
                    metrics['execution_time_seconds'] = hardware_data.get('average_execution_time', 0)
                    metrics['jobs_submitted'] = hardware_data.get('jobs_submitted', 0)
                    metrics['jobs_completed'] = hardware_data.get('jobs_completed', 0)
            
        except Exception as e:
            logger.error(f"Error collecting quantum metrics: {e}")
        
        return metrics
    
    def _collect_cost_metrics(self) -> Dict[str, Any]:
        """Collect cost-related metrics."""
        logger.info("Collecting cost metrics...")
        metrics = {}
        
        try:
            # Cost tracking data
            cost_file = Path('cost-tracking.json')
            if cost_file.exists():
                with open(cost_file, 'r') as f:
                    cost_data = json.load(f)
                    metrics['quantum_hardware_cost_usd'] = cost_data.get('quantum_hardware', 0)
                    metrics['cloud_infrastructure_cost_usd'] = cost_data.get('cloud_infrastructure', 0)
                    metrics['ci_cd_cost_usd'] = cost_data.get('ci_cd', 0)
                    metrics['total_monthly_cost_usd'] = cost_data.get('total_monthly', 0)
            
        except Exception as e:
            logger.error(f"Error collecting cost metrics: {e}")
        
        return metrics
    
    def _count_lines_of_code(self) -> int:
        """Count total lines of code in the project."""
        total_lines = 0
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.yaml', '.yml', '.json'}
        
        for root, dirs, files in os.walk('.'):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                'node_modules', '__pycache__', 'dist', 'build', 'coverage', 'venv'
            }]
            
            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for line in f if line.strip())
                    except Exception:
                        continue
        
        return total_lines
    
    def save_metrics(self, output_file: str = 'metrics-report.json') -> None:
        """Save collected metrics to a JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def send_to_monitoring(self, endpoint: str = None) -> None:
        """Send metrics to monitoring system."""
        if not endpoint:
            endpoint = os.getenv('METRICS_ENDPOINT')
        
        if not endpoint:
            logger.warning("No monitoring endpoint configured")
            return
        
        try:
            response = requests.post(
                endpoint,
                json=self.metrics,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            logger.info("Metrics sent to monitoring system successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending metrics to monitoring system: {e}")
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        report = []
        report.append("# Quantum DevOps CI Metrics Report")
        report.append(f"Generated at: {self.metrics.get('timestamp', 'Unknown')}")
        report.append("")
        
        # Repository metrics
        if 'repository' in self.metrics:
            repo_metrics = self.metrics['repository']
            report.append("## Repository Metrics")
            report.append(f"- Stars: {repo_metrics.get('stars', 'N/A')}")
            report.append(f"- Forks: {repo_metrics.get('forks', 'N/A')}")
            report.append(f"- Contributors: {repo_metrics.get('total_contributors', 'N/A')}")
            report.append(f"- Recent commits (30 days): {repo_metrics.get('commits_last_30_days', 'N/A')}")
            report.append("")
        
        # Development metrics
        if 'development' in self.metrics:
            dev_metrics = self.metrics['development']
            report.append("## Development Metrics")
            report.append(f"- Test coverage: {dev_metrics.get('test_coverage', 'N/A')}%")
            report.append(f"- Lines of code: {dev_metrics.get('lines_of_code', 'N/A')}")
            report.append(f"- ESLint errors: {dev_metrics.get('eslint_errors', 'N/A')}")
            report.append("")
        
        # Performance metrics
        if 'performance' in self.metrics:
            perf_metrics = self.metrics['performance']
            report.append("## Performance Metrics")
            report.append(f"- Average build time: {perf_metrics.get('average_build_time_seconds', 'N/A')}s")
            report.append(f"- Test success rate: {perf_metrics.get('test_success_rate', 'N/A')}%")
            report.append("")
        
        # Security metrics
        if 'security' in self.metrics:
            sec_metrics = self.metrics['security']
            report.append("## Security Metrics")
            report.append(f"- Critical vulnerabilities: {sec_metrics.get('critical_vulnerabilities', 'N/A')}")
            report.append(f"- High vulnerabilities: {sec_metrics.get('high_vulnerabilities', 'N/A')}")
            report.append("")
        
        # Quantum metrics
        if 'quantum' in self.metrics:
            quantum_metrics = self.metrics['quantum']
            report.append("## Quantum Metrics")
            report.append(f"- Circuits tested: {quantum_metrics.get('circuits_tested', 'N/A')}")
            report.append(f"- Average fidelity: {quantum_metrics.get('average_fidelity', 'N/A')}")
            report.append(f"- Gate count reduction: {quantum_metrics.get('gate_count_reduction', 'N/A')}%")
            report.append("")
        
        # Cost metrics
        if 'cost' in self.metrics:
            cost_metrics = self.metrics['cost']
            report.append("## Cost Metrics")
            report.append(f"- Quantum hardware cost: ${cost_metrics.get('quantum_hardware_cost_usd', 'N/A')}")
            report.append(f"- Total monthly cost: ${cost_metrics.get('total_monthly_cost_usd', 'N/A')}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for the metrics collection script."""
    collector = MetricsCollector()
    
    # Collect all metrics
    metrics = collector.collect_all_metrics()
    
    # Save to file
    collector.save_metrics()
    
    # Send to monitoring system (if configured)
    collector.send_to_monitoring()
    
    # Generate and print report
    report = collector.generate_report()
    print(report)
    
    # Save report to file
    with open('metrics-report.md', 'w') as f:
        f.write(report)
    
    logger.info("Metrics collection completed successfully")


if __name__ == '__main__':
    main()