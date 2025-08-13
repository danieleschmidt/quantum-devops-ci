"""
Predictive Analytics for Quantum DevOps - Generation 4 Intelligence.

This module implements advanced predictive models for cost forecasting,
performance optimization, and resource planning in quantum computing workflows.
"""

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

# Time series and ML libraries with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("scikit-learn not available - using statistical fallbacks")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available - limited time series features")

from .exceptions import PredictionError, ModelTrainingError, InsufficientDataError
from .validation import validate_inputs
from .security import requires_auth, audit_action
from .caching import CacheManager


class PredictionHorizon(Enum):
    """Time horizons for predictions."""
    HOUR = "1h"
    DAY = "1d" 
    WEEK = "1w"
    MONTH = "1m"
    QUARTER = "3m"
    YEAR = "1y"


class MetricType(Enum):
    """Types of metrics to predict."""
    COST = "cost"
    EXECUTION_TIME = "execution_time"
    QUEUE_TIME = "queue_time"
    SUCCESS_RATE = "success_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class TimeSeriesData:
    """Time series data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'metadata': self.metadata
        }


@dataclass 
class PredictionRequest:
    """Request for predictive analysis."""
    metric_type: MetricType
    horizon: PredictionHorizon
    context: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 0.95
    include_intervals: bool = True
    features: Optional[Dict[str, float]] = None


@dataclass
class PredictionResult:
    """Result of predictive analysis."""
    predicted_value: float
    confidence_intervals: Optional[Tuple[float, float]]
    prediction_horizon: PredictionHorizon
    model_confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    seasonal_components: Optional[Dict[str, float]] = None
    trend_analysis: Optional[str] = None
    
    def is_reliable(self) -> bool:
        """Check if prediction is reliable."""
        return self.model_confidence > 0.7


class QuantumCostPredictor:
    """Advanced cost prediction using multiple models and ensemble methods."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.models = {}
        self.training_data = {}
        self.feature_scalers = {}
        self.last_training = {}
        
    def add_historical_data(self, 
                          provider: str,
                          backend: str, 
                          timestamp: datetime,
                          cost: float,
                          features: Dict[str, float]):
        """Add historical cost data point."""
        key = f"{provider}_{backend}"
        
        if key not in self.training_data:
            self.training_data[key] = []
            
        data_point = {
            'timestamp': timestamp,
            'cost': cost,
            'features': features,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month
        }
        
        self.training_data[key].append(data_point)
        
        # Keep only recent data (last year)
        cutoff = datetime.now() - timedelta(days=365)
        self.training_data[key] = [
            dp for dp in self.training_data[key]
            if dp['timestamp'] > cutoff
        ]
        
    @requires_auth
    @audit_action("cost_model_training")
    def train_cost_models(self, provider: str, backend: str) -> Dict[str, float]:
        """Train cost prediction models."""
        key = f"{provider}_{backend}"
        
        if key not in self.training_data or len(self.training_data[key]) < 50:
            raise InsufficientDataError(f"Need at least 50 data points for {key}, have {len(self.training_data.get(key, []))}")
            
        data = self.training_data[key]
        
        # Prepare features and targets
        features = []
        targets = []
        
        for dp in data:
            feature_vector = [
                dp['features'].get('shots', 1000),
                dp['features'].get('circuit_depth', 10),
                dp['features'].get('gate_count', 50),
                dp['hour_of_day'],
                dp['day_of_week'],
                dp['month'],
                dp['features'].get('priority_multiplier', 1.0)
            ]
            features.append(feature_vector)
            targets.append(dp['cost'])
            
        if not ML_AVAILABLE:
            # Statistical fallback
            return self._train_statistical_cost_model(key, targets)
            
        X = np.array(features)
        y = np.array(targets)
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.feature_scalers[key] = scaler
        
        # Train ensemble of models
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        model_scores = {}
        trained_models = {}
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                model.fit(X_scaled, y)
                
                model_scores[name] = scores.mean()
                trained_models[name] = model
                
                logging.info(f"Trained {name} cost model for {key}: R² = {scores.mean():.3f}")
                
            except Exception as e:
                logging.warning(f"Failed to train {name} model: {e}")
                continue
                
        if not trained_models:
            raise ModelTrainingError(f"Failed to train any cost models for {key}")
            
        # Select best model or create ensemble
        if len(trained_models) > 1:
            self.models[key] = EnsembleModel(trained_models, model_scores)
        else:
            best_model = max(trained_models.items(), key=lambda x: model_scores[x[0]])
            self.models[key] = best_model[1]
            
        self.last_training[key] = datetime.now()
        return model_scores
        
    def _train_statistical_cost_model(self, key: str, targets: List[float]) -> Dict[str, float]:
        """Fallback statistical cost model."""
        mean_cost = sum(targets) / len(targets)
        std_cost = np.std(targets) if hasattr(np, 'std') else 0.0
        
        # Simple trend analysis
        if len(targets) > 10:
            recent_targets = targets[-10:]
            early_targets = targets[:10]
            trend = (sum(recent_targets) / len(recent_targets)) - (sum(early_targets) / len(early_targets))
        else:
            trend = 0.0
            
        self.models[key] = {
            'type': 'statistical',
            'mean': mean_cost,
            'std': std_cost,
            'trend': trend
        }
        
        return {'statistical': 0.5}  # Modest confidence
        
    @validate_inputs
    def predict_cost(self, 
                    provider: str,
                    backend: str,
                    shots: int,
                    circuit_depth: int,
                    gate_count: int,
                    target_time: Optional[datetime] = None,
                    confidence_level: float = 0.95) -> PredictionResult:
        """Predict cost for quantum job."""
        
        key = f"{provider}_{backend}"
        
        if key not in self.models:
            raise PredictionError(f"No trained model for {provider}_{backend}")
            
        target_time = target_time or datetime.now()
        
        # Prepare features
        features = np.array([[
            shots,
            circuit_depth, 
            gate_count,
            target_time.hour,
            target_time.weekday(),
            target_time.month,
            1.0  # Default priority multiplier
        ]])
        
        model = self.models[key]
        
        if isinstance(model, dict) and model.get('type') == 'statistical':
            # Statistical prediction
            base_prediction = model['mean']
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * target_time.hour / 24) if hasattr(np, 'sin') else 1.0
            trend_factor = 1.0 + model['trend'] * 0.01
            
            predicted_value = base_prediction * seasonal_factor * trend_factor * (shots / 1000)
            confidence = 0.6
            confidence_interval = (predicted_value * 0.8, predicted_value * 1.2)
            
        elif ML_AVAILABLE and key in self.feature_scalers:
            # ML prediction
            scaler = self.feature_scalers[key]
            features_scaled = scaler.transform(features)
            
            if isinstance(model, EnsembleModel):
                predicted_value, confidence = model.predict_with_confidence(features_scaled[0])
            else:
                predicted_value = model.predict(features_scaled)[0]
                confidence = 0.8  # Default confidence for single models
                
            # Calculate confidence intervals (simplified)
            margin = predicted_value * (1 - confidence_level) / 2
            confidence_interval = (predicted_value - margin, predicted_value + margin)
            
        else:
            raise PredictionError(f"Model not properly initialized for {key}")
            
        return PredictionResult(
            predicted_value=max(0.0, predicted_value),  # Ensure non-negative cost
            confidence_intervals=confidence_interval,
            prediction_horizon=PredictionHorizon.HOUR,
            model_confidence=confidence,
            trend_analysis=self._analyze_cost_trend(key, target_time)
        )
        
    def _analyze_cost_trend(self, key: str, target_time: datetime) -> str:
        """Analyze cost trends."""
        if key not in self.training_data:
            return "insufficient_data"
            
        recent_data = [
            dp for dp in self.training_data[key]
            if dp['timestamp'] > target_time - timedelta(days=30)
        ]
        
        if len(recent_data) < 10:
            return "insufficient_recent_data"
            
        costs = [dp['cost'] for dp in recent_data]
        if len(costs) < 2:
            return "stable"
            
        # Simple trend analysis
        recent_avg = sum(costs[-5:]) / len(costs[-5:])
        older_avg = sum(costs[:5]) / len(costs[:5])
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing" 
        else:
            return "stable"


class EnsembleModel:
    """Ensemble model combining multiple predictors."""
    
    def __init__(self, models: Dict[str, Any], scores: Dict[str, float]):
        self.models = models
        self.weights = self._calculate_weights(scores)
        
    def _calculate_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble weights from model scores."""
        total_score = sum(max(0.1, score) for score in scores.values())  # Ensure positive weights
        return {
            name: max(0.1, score) / total_score 
            for name, score in scores.items()
        }
        
    def predict_with_confidence(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction with confidence estimate."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict([features])[0]
                predictions.append(pred)
                weights.append(self.weights[name])
            except Exception as e:
                logging.warning(f"Model {name} prediction failed: {e}")
                continue
                
        if not predictions:
            raise PredictionError("All ensemble models failed")
            
        # Weighted average
        weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        
        # Confidence based on prediction variance and weights
        if len(predictions) > 1:
            variance = np.var(predictions) if hasattr(np, 'var') else 0.0
            confidence = max(0.1, 1.0 - variance / (weighted_pred + 1e-6))
        else:
            confidence = 0.7
            
        return weighted_pred, min(0.95, confidence)


class PerformancePredictor:
    """Predictor for quantum job performance metrics."""
    
    def __init__(self):
        self.historical_data = {}
        self.performance_models = {}
        
    def add_performance_data(self,
                           job_config: Dict[str, Any],
                           execution_time: float,
                           queue_time: float,
                           success_rate: float,
                           timestamp: Optional[datetime] = None):
        """Add historical performance data."""
        timestamp = timestamp or datetime.now()
        
        backend_key = f"{job_config.get('provider', 'unknown')}_{job_config.get('backend', 'unknown')}"
        
        if backend_key not in self.historical_data:
            self.historical_data[backend_key] = []
            
        data_point = {
            'timestamp': timestamp,
            'job_config': job_config,
            'execution_time': execution_time,
            'queue_time': queue_time, 
            'success_rate': success_rate,
            'shots': job_config.get('shots', 1000),
            'circuit_depth': job_config.get('circuit_depth', 10),
            'gate_count': job_config.get('gate_count', 50)
        }
        
        self.historical_data[backend_key].append(data_point)
        
    @validate_inputs
    def predict_performance(self,
                          job_config: Dict[str, Any],
                          horizon: PredictionHorizon = PredictionHorizon.HOUR) -> Dict[str, PredictionResult]:
        """Predict multiple performance metrics."""
        
        backend_key = f"{job_config.get('provider', 'unknown')}_{job_config.get('backend', 'unknown')}"
        
        if backend_key not in self.historical_data:
            return self._default_predictions(horizon)
            
        data = self.historical_data[backend_key]
        if len(data) < 10:
            return self._default_predictions(horizon)
            
        # Predict each metric
        results = {}
        
        # Execution time prediction
        execution_times = [dp['execution_time'] for dp in data]
        shots = job_config.get('shots', 1000)
        depth = job_config.get('circuit_depth', 10)
        
        # Simple linear model based on shots and depth
        base_time = sum(execution_times) / len(execution_times)
        scaling_factor = (shots / 1000) * (depth / 10)
        predicted_execution_time = base_time * scaling_factor
        
        results['execution_time'] = PredictionResult(
            predicted_value=predicted_execution_time,
            confidence_intervals=(predicted_execution_time * 0.8, predicted_execution_time * 1.2),
            prediction_horizon=horizon,
            model_confidence=0.7
        )
        
        # Queue time prediction (time-dependent)
        current_hour = datetime.now().hour
        queue_times = [dp['queue_time'] for dp in data]
        
        # Simple hourly pattern
        if hasattr(np, 'sin'):
            hourly_factor = 1.0 + 0.3 * np.sin(2 * np.pi * current_hour / 24)
        else:
            hourly_factor = 1.0
            
        avg_queue_time = sum(queue_times) / len(queue_times)
        predicted_queue_time = avg_queue_time * hourly_factor
        
        results['queue_time'] = PredictionResult(
            predicted_value=predicted_queue_time,
            confidence_intervals=(predicted_queue_time * 0.5, predicted_queue_time * 2.0),
            prediction_horizon=horizon,
            model_confidence=0.6
        )
        
        # Success rate prediction
        recent_data = data[-20:]  # Last 20 jobs
        success_rates = [dp['success_rate'] for dp in recent_data]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        results['success_rate'] = PredictionResult(
            predicted_value=avg_success_rate,
            confidence_intervals=(max(0.0, avg_success_rate - 0.1), min(1.0, avg_success_rate + 0.1)),
            prediction_horizon=horizon,
            model_confidence=0.8
        )
        
        return results
        
    def _default_predictions(self, horizon: PredictionHorizon) -> Dict[str, PredictionResult]:
        """Default predictions when no historical data available."""
        return {
            'execution_time': PredictionResult(
                predicted_value=60.0,  # 1 minute default
                confidence_intervals=(30.0, 120.0),
                prediction_horizon=horizon,
                model_confidence=0.3
            ),
            'queue_time': PredictionResult(
                predicted_value=300.0,  # 5 minutes default
                confidence_intervals=(60.0, 1800.0),
                prediction_horizon=horizon,
                model_confidence=0.2
            ),
            'success_rate': PredictionResult(
                predicted_value=0.95,  # 95% default
                confidence_intervals=(0.8, 0.99),
                prediction_horizon=horizon,
                model_confidence=0.4
            )
        }


class ResourcePlanningEngine:
    """Advanced resource planning using predictive analytics."""
    
    def __init__(self, 
                 cost_predictor: QuantumCostPredictor,
                 performance_predictor: PerformancePredictor):
        self.cost_predictor = cost_predictor
        self.performance_predictor = performance_predictor
        self.planning_cache = {}
        
    @requires_auth
    @audit_action("resource_planning")
    def create_resource_plan(self,
                           experiments: List[Dict[str, Any]],
                           budget_limit: float,
                           time_limit: timedelta,
                           optimization_goal: str = "cost") -> Dict[str, Any]:
        """Create optimal resource allocation plan."""
        
        plan_key = f"{len(experiments)}_{budget_limit}_{optimization_goal}_{datetime.now().strftime('%Y%m%d%H')}"
        
        # Check cache first
        if plan_key in self.planning_cache:
            return self.planning_cache[plan_key]
            
        provider_options = ['ibmq', 'aws_braket', 'google_quantum']
        backend_options = ['qasm_simulator', 'hardware_backend']
        
        optimal_assignments = []
        total_estimated_cost = 0.0
        total_estimated_time = 0.0
        
        for exp in experiments:
            best_assignment = None
            best_score = float('inf' if optimization_goal == 'cost' else '-inf')
            
            for provider in provider_options:
                for backend in backend_options:
                    try:
                        # Predict cost
                        cost_pred = self.cost_predictor.predict_cost(
                            provider=provider,
                            backend=backend,
                            shots=exp.get('shots', 1000),
                            circuit_depth=exp.get('circuit_depth', 10),
                            gate_count=exp.get('gate_count', 50)
                        )
                        
                        # Predict performance
                        job_config = {
                            'provider': provider,
                            'backend': backend,
                            'shots': exp.get('shots', 1000),
                            'circuit_depth': exp.get('circuit_depth', 10),
                            'gate_count': exp.get('gate_count', 50)
                        }
                        perf_preds = self.performance_predictor.predict_performance(job_config)
                        
                        # Calculate score based on optimization goal
                        if optimization_goal == 'cost':
                            score = cost_pred.predicted_value
                        elif optimization_goal == 'time':
                            exec_time = perf_preds['execution_time'].predicted_value
                            queue_time = perf_preds['queue_time'].predicted_value
                            score = -(exec_time + queue_time)  # Negative for minimization
                        else:  # balanced
                            normalized_cost = cost_pred.predicted_value / 100.0  # Normalize to similar scale
                            normalized_time = (perf_preds['execution_time'].predicted_value + 
                                             perf_preds['queue_time'].predicted_value) / 3600.0  # Hours
                            score = normalized_cost + normalized_time
                            
                        # Check constraints
                        if (total_estimated_cost + cost_pred.predicted_value > budget_limit or
                            total_estimated_time + perf_preds['execution_time'].predicted_value > time_limit.total_seconds()):
                            continue
                            
                        # Update best assignment
                        if ((optimization_goal == 'cost' and score < best_score) or
                            (optimization_goal == 'time' and score > best_score) or
                            (optimization_goal not in ['cost', 'time'] and score < best_score)):
                            
                            best_score = score
                            best_assignment = {
                                'experiment_id': exp.get('id', f"exp_{len(optimal_assignments)}"),
                                'provider': provider,
                                'backend': backend,
                                'predicted_cost': cost_pred.predicted_value,
                                'predicted_execution_time': perf_preds['execution_time'].predicted_value,
                                'predicted_queue_time': perf_preds['queue_time'].predicted_value,
                                'predicted_success_rate': perf_preds['success_rate'].predicted_value,
                                'cost_confidence': cost_pred.model_confidence,
                                'performance_confidence': perf_preds['execution_time'].model_confidence
                            }
                            
                    except Exception as e:
                        logging.warning(f"Failed to evaluate {provider}/{backend}: {e}")
                        continue
                        
            if best_assignment:
                optimal_assignments.append(best_assignment)
                total_estimated_cost += best_assignment['predicted_cost']
                total_estimated_time += best_assignment['predicted_execution_time']
                
        # Create comprehensive plan
        resource_plan = {
            'assignments': optimal_assignments,
            'total_experiments': len(experiments),
            'successfully_planned': len(optimal_assignments),
            'total_estimated_cost': total_estimated_cost,
            'total_estimated_time': total_estimated_time,
            'budget_utilization': total_estimated_cost / budget_limit if budget_limit > 0 else 0,
            'time_utilization': total_estimated_time / time_limit.total_seconds() if time_limit.total_seconds() > 0 else 0,
            'optimization_goal': optimization_goal,
            'created_at': datetime.now(),
            'recommendations': self._generate_recommendations(optimal_assignments, budget_limit, time_limit)
        }
        
        # Cache the plan
        self.planning_cache[plan_key] = resource_plan
        
        return resource_plan
        
    def _generate_recommendations(self, 
                                assignments: List[Dict[str, Any]], 
                                budget_limit: float,
                                time_limit: timedelta) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not assignments:
            recommendations.append("No feasible assignments found - consider increasing budget or time limits")
            return recommendations
            
        # Analyze provider distribution
        provider_counts = {}
        for assignment in assignments:
            provider = assignment['provider']
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
        if len(provider_counts) == 1:
            recommendations.append(f"All experiments assigned to {list(provider_counts.keys())[0]} - consider diversifying providers for better reliability")
            
        # Analyze confidence levels
        avg_cost_confidence = sum(a['cost_confidence'] for a in assignments) / len(assignments)
        if avg_cost_confidence < 0.7:
            recommendations.append("Low prediction confidence - consider gathering more historical data")
            
        # Budget utilization
        total_cost = sum(a['predicted_cost'] for a in assignments)
        if total_cost / budget_limit < 0.5:
            recommendations.append("Budget underutilized - consider running additional experiments or using higher-precision backends")
        elif total_cost / budget_limit > 0.9:
            recommendations.append("Near budget limit - monitor actual costs closely and have contingency plans")
            
        return recommendations


# Export main classes
__all__ = [
    'PredictionHorizon',
    'MetricType',
    'TimeSeriesData',
    'PredictionRequest', 
    'PredictionResult',
    'QuantumCostPredictor',
    'EnsembleModel',
    'PerformancePredictor',
    'ResourcePlanningEngine'
]