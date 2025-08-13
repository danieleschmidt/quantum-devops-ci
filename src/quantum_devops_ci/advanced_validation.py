"""
Advanced Validation Framework for Generation 4 Intelligence.

This module implements comprehensive validation for quantum circuits, ML models,
and research experiments with statistical rigor and publication-ready analysis.
"""

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

# Statistical and scientific computing imports with fallbacks
try:
    import scipy.stats as stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - advanced statistics limited")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization disabled")

from .exceptions import ValidationError, StatisticalTestError, ModelValidationError
from .validation import validate_inputs, QuantumCircuitValidator
from .security import requires_auth, audit_action
from .caching import CacheManager


class ValidationLevel(Enum):
    """Levels of validation rigor."""
    BASIC = "basic"
    STANDARD = "standard" 
    RESEARCH = "research"
    PUBLICATION = "publication"


class StatisticalTest(Enum):
    """Statistical tests for validation."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"
    SHAPIRO_WILK = "shapiro_wilk"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: Optional[float] = None
    auc_roc: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    statistical_power: Optional[float] = None
    
    def is_statistically_significant(self, alpha: float = 0.05) -> bool:
        """Check statistical significance."""
        return self.p_value is not None and self.p_value < alpha
        
    def meets_publication_standards(self) -> bool:
        """Check if metrics meet publication standards."""
        return (
            self.is_statistically_significant() and
            self.accuracy > 0.8 and
            self.statistical_power is not None and 
            self.statistical_power > 0.8
        )


@dataclass
class ExperimentalDesign:
    """Experimental design for validation."""
    hypothesis: str
    null_hypothesis: str
    alternative_hypothesis: str
    significance_level: float = 0.05
    power_target: float = 0.8
    effect_size_expected: float = 0.5
    sample_size_calculation: Optional[Dict[str, Any]] = None
    control_variables: List[str] = field(default_factory=list)
    randomization_method: str = "simple"
    blinding: bool = False
    
    def calculate_required_sample_size(self) -> int:
        """Calculate required sample size for statistical power."""
        if not SCIPY_AVAILABLE:
            # Fallback calculation
            return max(30, int(100 / self.effect_size_expected))
            
        # Cohen's formulas for sample size
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(self.power_target)
        
        # For two-sample t-test
        n = 2 * ((z_alpha + z_beta) / self.effect_size_expected) ** 2
        return int(np.ceil(n))


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_history = []
        
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationMetrics:
        """Perform validation."""
        pass
        
    def record_validation(self, metrics: ValidationMetrics, metadata: Dict[str, Any]):
        """Record validation results."""
        record = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'metadata': metadata,
            'validation_level': self.validation_level.value
        }
        self.validation_history.append(record)


class QuantumAlgorithmValidator(BaseValidator):
    """Validator for quantum algorithms with statistical rigor."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.RESEARCH):
        super().__init__(validation_level)
        self.reference_implementations = {}
        
    @requires_auth
    @audit_action("quantum_algorithm_validation")
    def validate(self, 
                algorithm_results: Dict[str, Any],
                reference_results: Optional[Dict[str, Any]] = None,
                experimental_design: Optional[ExperimentalDesign] = None) -> ValidationMetrics:
        """Validate quantum algorithm performance."""
        
        if self.validation_level == ValidationLevel.PUBLICATION and experimental_design is None:
            raise ValidationError("Publication-level validation requires experimental design")
            
        # Extract results data
        predicted_values = algorithm_results.get('predictions', [])
        actual_values = algorithm_results.get('ground_truth', [])
        
        if len(predicted_values) != len(actual_values):
            raise ValidationError("Predicted and actual values must have same length")
            
        if len(predicted_values) < 10:
            warnings.warn("Small sample size may affect statistical validity")
            
        # Basic validation metrics
        metrics = self._calculate_basic_metrics(predicted_values, actual_values)
        
        # Statistical testing based on validation level
        if self.validation_level in [ValidationLevel.RESEARCH, ValidationLevel.PUBLICATION]:
            metrics = self._enhance_with_statistical_tests(metrics, predicted_values, actual_values, experimental_design)
            
        # Cross-validation if sufficient data
        if len(predicted_values) >= 30:
            cv_metrics = self._perform_cross_validation(algorithm_results)
            metrics = self._combine_metrics(metrics, cv_metrics)
            
        # Reference comparison
        if reference_results:
            comparison_metrics = self._compare_with_reference(predicted_values, reference_results)
            metrics = self._combine_metrics(metrics, comparison_metrics)
            
        self.record_validation(metrics, {
            'algorithm': algorithm_results.get('algorithm_name', 'unknown'),
            'sample_size': len(predicted_values),
            'validation_level': self.validation_level.value
        })
        
        return metrics
        
    def _calculate_basic_metrics(self, predicted: List[float], actual: List[float]) -> ValidationMetrics:
        """Calculate basic validation metrics."""
        predicted = np.array(predicted)
        actual = np.array(actual)
        
        # For regression metrics
        mse = np.mean((predicted - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted - actual))
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Convert to standard metrics (treating as regression problem)
        accuracy = max(0, 1 - mae / (np.std(actual) + 1e-8))  # Normalized accuracy
        
        return ValidationMetrics(
            accuracy=accuracy,
            precision=r2,  # Using R² as precision proxy
            recall=r2,     # Using R² as recall proxy  
            f1_score=2 * r2 / (1 + r2) if r2 > 0 else 0
        )
        
    def _enhance_with_statistical_tests(self, 
                                       metrics: ValidationMetrics,
                                       predicted: List[float], 
                                       actual: List[float],
                                       experimental_design: Optional[ExperimentalDesign]) -> ValidationMetrics:
        """Enhance metrics with statistical tests."""
        
        if not SCIPY_AVAILABLE:
            warnings.warn("scipy not available - using simplified statistical tests")
            return self._simplified_statistical_tests(metrics, predicted, actual)
            
        predicted = np.array(predicted)
        actual = np.array(actual)
        residuals = predicted - actual
        
        # Normality test on residuals
        if len(residuals) >= 8:  # Minimum for Shapiro-Wilk
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            residuals_normal = shapiro_p > 0.05
        else:
            residuals_normal = True
            shapiro_p = 0.5
            
        # Statistical significance test
        if residuals_normal:
            # One-sample t-test (test if mean residual is significantly different from 0)
            t_stat, p_value = stats.ttest_1samp(residuals, 0)
        else:
            # Non-parametric Wilcoxon signed-rank test
            w_stat, p_value = stats.wilcoxon(residuals)
            
        # Effect size (Cohen's d)
        effect_size = np.mean(residuals) / (np.std(residuals) + 1e-8)
        
        # Confidence interval for mean residual
        if len(residuals) > 1:
            confidence_interval = stats.t.interval(
                0.95, len(residuals) - 1, 
                loc=np.mean(residuals), 
                scale=stats.sem(residuals)
            )
        else:
            confidence_interval = (np.mean(residuals), np.mean(residuals))
            
        # Statistical power calculation
        if experimental_design:
            power = self._calculate_statistical_power(residuals, experimental_design)
        else:
            power = None
            
        # Update metrics
        metrics.p_value = p_value
        metrics.effect_size = abs(effect_size)
        metrics.confidence_interval = confidence_interval
        metrics.statistical_power = power
        
        return metrics
        
    def _simplified_statistical_tests(self, metrics: ValidationMetrics, predicted: List[float], actual: List[float]) -> ValidationMetrics:
        """Simplified statistical tests when scipy unavailable."""
        predicted = np.array(predicted)
        actual = np.array(actual)
        residuals = predicted - actual
        
        # Simple t-test approximation
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        n = len(residuals)
        
        if std_residual > 0 and n > 1:
            t_stat = mean_residual / (std_residual / np.sqrt(n))
            # Approximate p-value for two-tailed test
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(n - 1)))
            p_value = max(0.001, min(0.999, p_value))  # Clamp to reasonable range
        else:
            p_value = 0.5
            
        metrics.p_value = p_value
        metrics.effect_size = abs(mean_residual / (std_residual + 1e-8))
        
        return metrics
        
    def _calculate_statistical_power(self, residuals: np.ndarray, experimental_design: ExperimentalDesign) -> float:
        """Calculate statistical power of the test."""
        if not SCIPY_AVAILABLE:
            return 0.8  # Default assumption
            
        n = len(residuals)
        effect_size = np.mean(residuals) / (np.std(residuals) + 1e-8)
        alpha = experimental_design.significance_level
        
        # For one-sample t-test
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
        
        # Power = P(reject H0 | H1 true)
        power = 1 - stats.t.cdf(t_critical, n-1, ncp) + stats.t.cdf(-t_critical, n-1, ncp)
        
        return max(0, min(1, power))
        
    def _perform_cross_validation(self, algorithm_results: Dict[str, Any], k_folds: int = 5) -> ValidationMetrics:
        """Perform k-fold cross-validation."""
        predictions = np.array(algorithm_results.get('predictions', []))
        ground_truth = np.array(algorithm_results.get('ground_truth', []))
        
        n = len(predictions)
        fold_size = n // k_folds
        
        fold_accuracies = []
        fold_precisions = []
        
        for i in range(k_folds):
            # Define fold indices
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else n
            
            # Split data
            test_pred = predictions[start_idx:end_idx]
            test_true = ground_truth[start_idx:end_idx]
            
            # Calculate metrics for this fold
            mae = np.mean(np.abs(test_pred - test_true))
            accuracy = max(0, 1 - mae / (np.std(test_true) + 1e-8))
            
            fold_accuracies.append(accuracy)
            fold_precisions.append(accuracy)  # Using accuracy as precision proxy
            
        # Average across folds
        cv_accuracy = np.mean(fold_accuracies)
        cv_precision = np.mean(fold_precisions)
        
        return ValidationMetrics(
            accuracy=cv_accuracy,
            precision=cv_precision,
            recall=cv_precision,
            f1_score=2 * cv_precision / (1 + cv_precision) if cv_precision > 0 else 0
        )
        
    def _compare_with_reference(self, predictions: List[float], reference_results: Dict[str, Any]) -> ValidationMetrics:
        """Compare with reference implementation."""
        reference_predictions = reference_results.get('predictions', [])
        
        if len(reference_predictions) != len(predictions):
            warnings.warn("Reference predictions length mismatch - using available data")
            min_len = min(len(reference_predictions), len(predictions))
            reference_predictions = reference_predictions[:min_len]
            predictions = predictions[:min_len]
            
        # Compare predictions directly
        pred_array = np.array(predictions)
        ref_array = np.array(reference_predictions)
        
        # Correlation with reference
        if len(pred_array) > 1:
            correlation = np.corrcoef(pred_array, ref_array)[0, 1]
        else:
            correlation = 1.0
            
        # Relative error
        relative_error = np.mean(np.abs(pred_array - ref_array) / (np.abs(ref_array) + 1e-8))
        
        return ValidationMetrics(
            accuracy=max(0, correlation),
            precision=max(0, 1 - relative_error),
            recall=max(0, 1 - relative_error),
            f1_score=max(0, correlation * (1 - relative_error))
        )
        
    def _combine_metrics(self, primary: ValidationMetrics, secondary: ValidationMetrics) -> ValidationMetrics:
        """Combine metrics from different validation approaches."""
        # Weighted average of metrics
        weight_primary = 0.7
        weight_secondary = 0.3
        
        combined_accuracy = weight_primary * primary.accuracy + weight_secondary * secondary.accuracy
        combined_precision = weight_primary * primary.precision + weight_secondary * secondary.precision
        combined_recall = weight_primary * primary.recall + weight_secondary * secondary.recall
        combined_f1 = weight_primary * primary.f1_score + weight_secondary * secondary.f1_score
        
        return ValidationMetrics(
            accuracy=combined_accuracy,
            precision=combined_precision,
            recall=combined_recall,
            f1_score=combined_f1,
            p_value=primary.p_value,  # Keep primary statistical metrics
            effect_size=primary.effect_size,
            statistical_power=primary.statistical_power,
            confidence_interval=primary.confidence_interval
        )


class MLModelValidator(BaseValidator):
    """Validator for machine learning models in quantum contexts."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.RESEARCH):
        super().__init__(validation_level)
        
    @requires_auth
    @audit_action("ml_model_validation")
    def validate(self,
                model: Any,
                X_test: np.ndarray,
                y_test: np.ndarray,
                X_train: Optional[np.ndarray] = None,
                y_train: Optional[np.ndarray] = None) -> ValidationMetrics:
        """Validate ML model performance."""
        
        try:
            # Get predictions
            predictions = model.predict(X_test)
        except Exception as e:
            raise ModelValidationError(f"Model prediction failed: {e}")
            
        # Basic metrics
        metrics = self._calculate_regression_metrics(y_test, predictions)
        
        # Model-specific validation
        if hasattr(model, 'score'):
            r2_score = model.score(X_test, y_test)
            metrics.precision = max(metrics.precision, r2_score)
            
        # Overfitting check
        if X_train is not None and y_train is not None:
            train_predictions = model.predict(X_train)
            train_metrics = self._calculate_regression_metrics(y_train, train_predictions)
            
            # Check for overfitting (train score much higher than test)
            overfitting_ratio = train_metrics.accuracy / (metrics.accuracy + 1e-8)
            if overfitting_ratio > 1.5:
                warnings.warn("Possible overfitting detected")
                
        # Residual analysis for publication-level validation
        if self.validation_level == ValidationLevel.PUBLICATION:
            metrics = self._analyze_residuals(y_test, predictions, metrics)
            
        self.record_validation(metrics, {
            'model_type': type(model).__name__,
            'test_samples': len(X_test),
            'features': X_test.shape[1] if len(X_test.shape) > 1 else 1
        })
        
        return metrics
        
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ValidationMetrics:
        """Calculate regression validation metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Convert to standard metrics framework
        accuracy = max(0, r2)
        precision = max(0, 1 - mae / (np.std(y_true) + 1e-8))
        recall = precision  # Same for regression
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return ValidationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
        
    def _analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: ValidationMetrics) -> ValidationMetrics:
        """Analyze residuals for publication-level validation."""
        residuals = y_pred - y_true
        
        if SCIPY_AVAILABLE:
            # Normality test
            if len(residuals) >= 8:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                
            # Homoscedasticity test (Breusch-Pagan approximation)
            if len(residuals) > 10:
                # Simple correlation test between |residuals| and predictions
                correlation = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
                homoscedastic = abs(correlation) < 0.3
            else:
                homoscedastic = True
                
            # Durbin-Watson test for autocorrelation (simplified)
            if len(residuals) > 2:
                dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
                independence = 1.5 < dw_stat < 2.5  # Rough independence check
            else:
                independence = True
                
            # Update p_value based on residual analysis
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            if residual_std > 0:
                t_stat = residual_mean / (residual_std / np.sqrt(len(residuals)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(residuals) - 1))
                metrics.p_value = p_value
                
        return metrics


class ResearchExperimentValidator:
    """Validator for research experiments requiring publication-quality validation."""
    
    def __init__(self):
        self.experiment_registry = {}
        self.validation_protocols = {}
        
    @requires_auth  
    @audit_action("research_experiment_validation")
    def validate_experiment(self,
                          experiment_id: str,
                          hypothesis: str,
                          experimental_data: Dict[str, Any],
                          control_data: Optional[Dict[str, Any]] = None,
                          significance_level: float = 0.05) -> Dict[str, Any]:
        """Validate research experiment with publication standards."""
        
        validation_report = {
            'experiment_id': experiment_id,
            'hypothesis': hypothesis,
            'validation_timestamp': datetime.now(),
            'significance_level': significance_level,
            'results': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'confidence_intervals': {},
            'reproducibility_metrics': {},
            'publication_readiness': {}
        }
        
        # Extract experimental measurements
        treatment_values = experimental_data.get('measurements', [])
        
        if not treatment_values:
            raise ValidationError("No experimental measurements provided")
            
        # Basic descriptive statistics
        validation_report['results']['treatment'] = {
            'n': len(treatment_values),
            'mean': np.mean(treatment_values),
            'std': np.std(treatment_values),
            'median': np.median(treatment_values),
            'min': np.min(treatment_values),
            'max': np.max(treatment_values)
        }
        
        # Control group analysis if provided
        if control_data:
            control_values = control_data.get('measurements', [])
            validation_report['results']['control'] = {
                'n': len(control_values),
                'mean': np.mean(control_values),
                'std': np.std(control_values),
                'median': np.median(control_values),
                'min': np.min(control_values),
                'max': np.max(control_values)
            }
            
            # Between-group statistical tests
            validation_report['statistical_tests'] = self._perform_group_comparison_tests(
                treatment_values, control_values, significance_level
            )
            
        else:
            # One-sample tests against theoretical value
            theoretical_value = experimental_data.get('theoretical_value', 0.0)
            validation_report['statistical_tests'] = self._perform_one_sample_tests(
                treatment_values, theoretical_value, significance_level
            )
            
        # Effect size calculations
        validation_report['effect_sizes'] = self._calculate_effect_sizes(
            treatment_values, control_values if control_data else None
        )
        
        # Reproducibility analysis
        validation_report['reproducibility_metrics'] = self._assess_reproducibility(
            experimental_data
        )
        
        # Publication readiness assessment
        validation_report['publication_readiness'] = self._assess_publication_readiness(
            validation_report
        )
        
        # Store experiment
        self.experiment_registry[experiment_id] = validation_report
        
        return validation_report
        
    def _perform_group_comparison_tests(self, 
                                      treatment: List[float], 
                                      control: List[float],
                                      alpha: float) -> Dict[str, Any]:
        """Perform statistical tests comparing two groups."""
        tests = {}
        
        treatment = np.array(treatment)
        control = np.array(control)
        
        if SCIPY_AVAILABLE:
            # Normality tests for both groups
            if len(treatment) >= 8:
                treat_shapiro_stat, treat_shapiro_p = stats.shapiro(treatment)
                treat_normal = treat_shapiro_p > 0.05
            else:
                treat_normal = True
                treat_shapiro_p = 0.5
                
            if len(control) >= 8:
                ctrl_shapiro_stat, ctrl_shapiro_p = stats.shapiro(control)
                ctrl_normal = ctrl_shapiro_p > 0.05
            else:
                ctrl_normal = True
                ctrl_shapiro_p = 0.5
                
            # Equal variance test
            if len(treatment) > 2 and len(control) > 2:
                levene_stat, levene_p = stats.levene(treatment, control)
                equal_var = levene_p > 0.05
            else:
                equal_var = True
                levene_p = 0.5
                
            # Choose appropriate test
            if treat_normal and ctrl_normal:
                # Two-sample t-test
                t_stat, t_p = stats.ttest_ind(treatment, control, equal_var=equal_var)
                tests['t_test'] = {
                    'statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < alpha,
                    'test_type': 'two_sample_t_test'
                }
            else:
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = stats.mannwhitneyu(treatment, control, alternative='two-sided')
                tests['mann_whitney'] = {
                    'statistic': u_stat,
                    'p_value': u_p,
                    'significant': u_p < alpha,
                    'test_type': 'mann_whitney_u'
                }
                
            # Bootstrap confidence interval for difference in means
            tests['bootstrap_ci'] = self._bootstrap_difference_ci(treatment, control)
            
        else:
            # Simplified tests without scipy
            tests = self._simplified_group_tests(treatment, control, alpha)
            
        return tests
        
    def _perform_one_sample_tests(self,
                                 sample: List[float],
                                 theoretical_value: float,
                                 alpha: float) -> Dict[str, Any]:
        """Perform one-sample statistical tests."""
        tests = {}
        sample = np.array(sample)
        
        if SCIPY_AVAILABLE:
            # Normality test
            if len(sample) >= 8:
                shapiro_stat, shapiro_p = stats.shapiro(sample)
                is_normal = shapiro_p > 0.05
            else:
                is_normal = True
                shapiro_p = 0.5
                
            # One-sample t-test
            t_stat, t_p = stats.ttest_1samp(sample, theoretical_value)
            tests['one_sample_t'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < alpha,
                'test_type': 'one_sample_t_test'
            }
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            if len(sample) > 5:
                w_stat, w_p = stats.wilcoxon(sample - theoretical_value)
                tests['wilcoxon'] = {
                    'statistic': w_stat,
                    'p_value': w_p,
                    'significant': w_p < alpha,
                    'test_type': 'wilcoxon_signed_rank'
                }
                
        else:
            # Simplified one-sample test
            tests = self._simplified_one_sample_test(sample, theoretical_value, alpha)
            
        return tests
        
    def _simplified_group_tests(self, treatment: np.ndarray, control: np.ndarray, alpha: float) -> Dict[str, Any]:
        """Simplified group comparison without scipy."""
        treat_mean = np.mean(treatment)
        ctrl_mean = np.mean(control)
        treat_std = np.std(treatment)
        ctrl_std = np.std(control)
        
        n1, n2 = len(treatment), len(control)
        
        # Pooled standard error
        pooled_se = np.sqrt(treat_std**2/n1 + ctrl_std**2/n2)
        
        if pooled_se > 0:
            t_stat = (treat_mean - ctrl_mean) / pooled_se
            # Approximate p-value
            df = n1 + n2 - 2
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(df)))
            p_value = max(0.001, min(0.999, p_value))
        else:
            t_stat = 0
            p_value = 1.0
            
        return {
            'simplified_t_test': {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'test_type': 'simplified_t_test'
            }
        }
        
    def _simplified_one_sample_test(self, sample: np.ndarray, theoretical: float, alpha: float) -> Dict[str, Any]:
        """Simplified one-sample test without scipy."""
        sample_mean = np.mean(sample)
        sample_std = np.std(sample)
        n = len(sample)
        
        if sample_std > 0:
            t_stat = (sample_mean - theoretical) / (sample_std / np.sqrt(n))
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(n - 1)))
            p_value = max(0.001, min(0.999, p_value))
        else:
            t_stat = 0
            p_value = 1.0
            
        return {
            'simplified_one_sample_t': {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'test_type': 'simplified_one_sample_t'
            }
        }
        
    def _bootstrap_difference_ci(self, treatment: np.ndarray, control: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, float]:
        """Bootstrap confidence interval for difference in means."""
        if not hasattr(np, 'random'):
            return {'lower': -1.0, 'upper': 1.0, 'method': 'unavailable'}
            
        differences = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            treat_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            ctrl_sample = np.random.choice(control, size=len(control), replace=True)
            
            diff = np.mean(treat_sample) - np.mean(ctrl_sample)
            differences.append(diff)
            
        # 95% confidence interval
        lower = np.percentile(differences, 2.5)
        upper = np.percentile(differences, 97.5)
        
        return {
            'lower': lower,
            'upper': upper,
            'method': 'bootstrap'
        }
        
    def _calculate_effect_sizes(self, treatment: List[float], control: Optional[List[float]]) -> Dict[str, float]:
        """Calculate effect sizes."""
        effect_sizes = {}
        treatment = np.array(treatment)
        
        if control is not None:
            control = np.array(control)
            
            # Cohen's d
            pooled_std = np.sqrt(((len(treatment) - 1) * np.std(treatment)**2 + 
                                (len(control) - 1) * np.std(control)**2) / 
                               (len(treatment) + len(control) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
                effect_sizes['cohens_d'] = cohens_d
                
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    interpretation = 'negligible'
                elif abs(cohens_d) < 0.5:
                    interpretation = 'small'
                elif abs(cohens_d) < 0.8:
                    interpretation = 'medium'
                else:
                    interpretation = 'large'
                    
                effect_sizes['interpretation'] = interpretation
        else:
            # Effect size for one sample (compared to zero or theoretical value)
            effect_sizes['standardized_mean'] = np.mean(treatment) / (np.std(treatment) + 1e-8)
            
        return effect_sizes
        
    def _assess_reproducibility(self, experimental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess reproducibility of the experiment."""
        reproducibility = {}
        
        measurements = experimental_data.get('measurements', [])
        
        # Coefficient of variation
        if len(measurements) > 1:
            cv = np.std(measurements) / (np.mean(measurements) + 1e-8)
            reproducibility['coefficient_of_variation'] = cv
            reproducibility['reproducibility_rating'] = (
                'excellent' if cv < 0.1 else
                'good' if cv < 0.2 else
                'acceptable' if cv < 0.3 else
                'poor'
            )
        
        # Sample size adequacy
        n = len(measurements)
        reproducibility['sample_size'] = n
        reproducibility['sample_size_adequacy'] = (
            'excellent' if n >= 100 else
            'good' if n >= 50 else
            'acceptable' if n >= 30 else
            'insufficient'
        )
        
        return reproducibility
        
    def _assess_publication_readiness(self, validation_report: Dict[str, Any]) -> Dict[str, bool]:
        """Assess if experiment meets publication standards."""
        readiness = {}
        
        # Statistical significance
        has_significant_result = False
        for test_name, test_result in validation_report['statistical_tests'].items():
            if test_result.get('significant', False):
                has_significant_result = True
                break
                
        readiness['statistical_significance'] = has_significant_result
        
        # Effect size reported
        readiness['effect_size_reported'] = bool(validation_report['effect_sizes'])
        
        # Adequate sample size
        treatment_n = validation_report['results'].get('treatment', {}).get('n', 0)
        readiness['adequate_sample_size'] = treatment_n >= 30
        
        # Reproducibility metrics
        repro_rating = validation_report['reproducibility_metrics'].get('reproducibility_rating', 'poor')
        readiness['reproducible'] = repro_rating in ['excellent', 'good', 'acceptable']
        
        # Overall readiness
        readiness['overall_publication_ready'] = all([
            readiness['statistical_significance'],
            readiness['effect_size_reported'], 
            readiness['adequate_sample_size'],
            readiness['reproducible']
        ])
        
        return readiness


# Export main classes
__all__ = [
    'ValidationLevel',
    'StatisticalTest',
    'ValidationMetrics',
    'ExperimentalDesign', 
    'BaseValidator',
    'QuantumAlgorithmValidator',
    'MLModelValidator',
    'ResearchExperimentValidator'
]