"""
Classification Engine for Otto Group Product Classification.

Complete implementation of model training, evaluation, calibration, and ensemble.
This is the core module for the multi-class classification challenge.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import json
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    """
    Handles training of individual models with cross-validation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.cv_scores = {}
        
    def train_random_forest(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           n_estimators: int = 300,
                           max_depth: Optional[int] = 20,
                           random_state: int = 42,
                           **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            **kwargs: Additional parameters for RandomForestClassifier
        
        Returns:
            Trained RandomForestClassifier
        """
        self.logger.info(f"Training Random Forest (n_estimators={n_estimators}, "
                        f"max_depth={max_depth})")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
        
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        self.logger.info("Random Forest training completed")
        return rf
    
    def train_xgboost(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     n_estimators: int = 300,
                     learning_rate: float = 0.1,
                     max_depth: int = 6,
                     random_state: int = 42,
                     **kwargs) -> Any:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
            **kwargs: Additional parameters for XGBClassifier
        
        Returns:
            Trained XGBClassifier
        """
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available, skipping")
            return None
        
        self.logger.info(f"Training XGBoost (n_estimators={n_estimators}, "
                        f"lr={learning_rate}, max_depth={max_depth})")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            tree_method='hist',
            n_jobs=-1,
            **kwargs
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        self.logger.info("XGBoost training completed")
        return xgb_model
    
    def train_mlp(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 hidden_layers: Tuple[int, ...] = (256, 128),
                 max_iter: int = 200,
                 random_state: int = 42,
                 **kwargs) -> MLPClassifier:
        """
        Train Multi-Layer Perceptron (Neural Network) classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hidden_layers: Tuple of hidden layer sizes
            max_iter: Maximum iterations
            random_state: Random seed
            **kwargs: Additional parameters for MLPClassifier
        
        Returns:
            Trained MLPClassifier
        """
        self.logger.info(f"Training MLP (hidden_layers={hidden_layers}, "
                        f"max_iter={max_iter})")
        
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            **kwargs
        )
        
        mlp.fit(X_train, y_train)
        self.models['mlp'] = mlp
        
        self.logger.info(f"MLP training completed (converged in {mlp.n_iter_} iterations)")
        return mlp
    
    def train_logistic(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      random_state: int = 42,
                      **kwargs) -> LogisticRegression:
        """
        Train Logistic Regression (baseline model).
        
        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random seed
            **kwargs: Additional parameters for LogisticRegression
        
        Returns:
            Trained LogisticRegression
        """
        self.logger.info("Training Logistic Regression (baseline)")
        
        lr = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
        
        lr.fit(X_train, y_train)
        self.models['logistic'] = lr
        
        self.logger.info("Logistic Regression training completed")
        return lr
    
    def cross_validate_model(self,
                            model: Any,
                            X: np.ndarray,
                            y: np.ndarray,
                            cv: int = 5,
                            model_name: str = 'model') -> Dict[str, float]:
        """
        Perform cross-validation and calculate metrics.
        
        Args:
            model: Trained model
            X: Features
            y: Labels
            cv: Number of CV folds
            model_name: Name for logging
        
        Returns:
            Dictionary with CV scores
        """
        self.logger.info(f"Cross-validating {model_name} ({cv} folds)")
        
        # Log loss (lower is better)
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='neg_log_loss',
            n_jobs=-1
        )
        
        cv_results = {
            'mean_logloss': -scores.mean(),
            'std_logloss': scores.std(),
            'scores': (-scores).tolist()
        }
        
        self.cv_scores[model_name] = cv_results
        
        self.logger.info(f"{model_name} CV Log Loss: "
                        f"{cv_results['mean_logloss']:.4f} ± {cv_results['std_logloss']:.4f}")
        
        return cv_results
    
    def get_models(self) -> Dict[str, Any]:
        """Get all trained models."""
        return self.models
    
    def get_cv_scores(self) -> Dict[str, Dict]:
        """Get all cross-validation scores."""
        return self.cv_scores


class ModelEvaluator:
    """
    Evaluates model performance with various metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ModelEvaluator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.evaluation_results = {}
    
    def evaluate(self,
                model: Any,
                X_val: np.ndarray,
                y_val: np.ndarray,
                model_name: str = 'model') -> Dict[str, Any]:
        """
        Evaluate model on validation set.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            model_name: Name for logging
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        logloss = log_loss(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        results = {
            'logloss': float(logloss),
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'n_samples': len(y_val)
        }
        
        self.evaluation_results[model_name] = results
        
        self.logger.info(f"{model_name} - Log Loss: {logloss:.4f}, Accuracy: {accuracy:.4f}")
        
        return results
    
    def compare_models(self, results: Dict[str, Dict]) -> str:
        """
        Compare multiple models and return summary.
        
        Args:
            results: Dictionary of model evaluation results
        
        Returns:
            Formatted comparison string
        """
        comparison = "\n" + "="*60 + "\n"
        comparison += "MODEL COMPARISON\n"
        comparison += "="*60 + "\n\n"
        
        # Sort by log loss (lower is better)
        sorted_models = sorted(results.items(), key=lambda x: x[1]['logloss'])
        
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            comparison += f"{rank}. {name.upper()}\n"
            comparison += f"   Log Loss: {metrics['logloss']:.4f}\n"
            comparison += f"   Accuracy: {metrics['accuracy']:.4f}\n\n"
        
        comparison += "="*60 + "\n"
        
        return comparison
    
    def get_results(self) -> Dict[str, Dict]:
        """Get all evaluation results."""
        return self.evaluation_results


class ModelCalibrator:
    """
    Calibrates model probabilities using Platt scaling or Isotonic regression.
    """
    
    def __init__(self, 
                 method: str = 'isotonic',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ModelCalibrator.
        
        Args:
            method: Calibration method ('sigmoid' or 'isotonic')
            logger: Optional logger instance
        """
        self.method = method
        self.logger = logger or logging.getLogger(__name__)
        self.calibrated_models = {}
    
    def calibrate(self,
                 model: Any,
                 X_cal: np.ndarray,
                 y_cal: np.ndarray,
                 model_name: str = 'model',
                 cv: int = 3) -> CalibratedClassifierCV:
        """
        Calibrate a trained model.
        
        Args:
            model: Trained model to calibrate
            X_cal: Calibration features
            y_cal: Calibration labels
            model_name: Name for logging
            cv: Number of CV folds for calibration
        
        Returns:
            Calibrated model
        """
        self.logger.info(f"Calibrating {model_name} using {self.method} method (cv={cv})")
        
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=cv
        )
        
        calibrated.fit(X_cal, y_cal)
        self.calibrated_models[model_name] = calibrated
        
        # Compare before/after
        y_pred_before = model.predict_proba(X_cal)
        y_pred_after = calibrated.predict_proba(X_cal)
        
        logloss_before = log_loss(y_cal, y_pred_before)
        logloss_after = log_loss(y_cal, y_pred_after)
        
        improvement = logloss_before - logloss_after
        self.logger.info(f"Calibration: {logloss_before:.4f} -> {logloss_after:.4f} "
                        f"(improvement: {improvement:.4f})")
        
        return calibrated
    
    def get_calibrated_models(self) -> Dict[str, Any]:
        """Get all calibrated models."""
        return self.calibrated_models


class EnsembleBuilder:
    """
    Creates ensemble models from multiple trained classifiers.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize EnsembleBuilder.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.ensemble = None
    
    def create_voting_ensemble(self,
                              models: List[Tuple[str, Any]],
                              voting: str = 'soft',
                              weights: Optional[List[float]] = None) -> VotingClassifier:
        """
        Create a voting ensemble from multiple models.
        
        Args:
            models: List of (name, model) tuples
            voting: 'soft' for probability averaging, 'hard' for majority vote
            weights: Optional weights for each model
        
        Returns:
            VotingClassifier ensemble
        """
        self.logger.info(f"Creating {voting} voting ensemble with {len(models)} models")
        
        if weights:
            self.logger.info(f"Using weights: {weights}")
        
        self.ensemble = VotingClassifier(
            estimators=models,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        self.logger.info("Ensemble created successfully")
        return self.ensemble
    
    def create_weighted_average(self,
                               models: Dict[str, Any],
                               X_val: np.ndarray,
                               y_val: np.ndarray) -> Tuple[List[Tuple[str, Any]], List[float]]:
        """
        Create weighted average ensemble based on validation performance.
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Tuple of (model_list, weights)
        """
        self.logger.info("Calculating optimal weights based on validation log loss")
        
        model_list = []
        logloss_scores = []
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_val)
            logloss = log_loss(y_val, y_pred_proba)
            logloss_scores.append(logloss)
            model_list.append((name, model))
            self.logger.info(f"{name}: {logloss:.4f}")
        
        # Convert log loss to weights (inverse)
        # Lower log loss = higher weight
        weights = [1.0 / score for score in logloss_scores]
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        self.logger.info(f"Calculated weights: {dict(zip([m[0] for m in model_list], weights))}")
        
        return model_list, weights
    
    def get_ensemble(self) -> Optional[VotingClassifier]:
        """Get the created ensemble."""
        return self.ensemble


class ClassificationEngine:
    """
    Main classification engine that orchestrates the entire pipeline.
    """
    
    def __init__(self,
                 use_calibration: bool = True,
                 calibration_method: str = 'isotonic',
                 use_ensemble: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ClassificationEngine.
        
        Args:
            use_calibration: Whether to calibrate probabilities
            calibration_method: Calibration method ('sigmoid' or 'isotonic')
            use_ensemble: Whether to create ensemble
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.use_ensemble = use_ensemble
        
        self.trainer = ModelTrainer(logger=logger)
        self.evaluator = ModelEvaluator(logger=logger)
        self.calibrator = ModelCalibrator(method=calibration_method, logger=logger)
        self.ensemble_builder = EnsembleBuilder(logger=logger)
        
        self.best_model = None
        self.best_model_name = None
        self.final_model = None
    
    def train_all_models(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        models_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_config: Optional configuration for models
        
        Returns:
            Dictionary of trained models
        """
        self.logger.info("=" * 60)
        self.logger.info("TRAINING ALL MODELS")
        self.logger.info("=" * 60)
        
        if models_config is None:
            models_config = {
                'random_forest': {'n_estimators': 300, 'max_depth': 20},
                'xgboost': {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 6},
                'mlp': {'hidden_layers': (256, 128), 'max_iter': 200},
                'logistic': {}
            }
        
        # Train Random Forest
        if 'random_forest' in models_config:
            self.trainer.train_random_forest(X_train, y_train, **models_config['random_forest'])
        
        # Train XGBoost (if available)
        if 'xgboost' in models_config and XGBOOST_AVAILABLE:
            self.trainer.train_xgboost(X_train, y_train, **models_config['xgboost'])
        
        # Train MLP
        if 'mlp' in models_config:
            self.trainer.train_mlp(X_train, y_train, **models_config['mlp'])
        
        # Train Logistic Regression (baseline)
        if 'logistic' in models_config:
            self.trainer.train_logistic(X_train, y_train, **models_config['logistic'])
        
        return self.trainer.get_models()
    
    def evaluate_all_models(self,
                           X_val: np.ndarray,
                           y_val: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all trained models.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("=" * 60)
        self.logger.info("EVALUATING ALL MODELS")
        self.logger.info("=" * 60)
        
        models = self.trainer.get_models()
        results = {}
        
        for name, model in models.items():
            results[name] = self.evaluator.evaluate(model, X_val, y_val, model_name=name)
        
        # Print comparison
        comparison = self.evaluator.compare_models(results)
        self.logger.info(comparison)
        
        # Find best model
        best_name = min(results.items(), key=lambda x: x[1]['logloss'])[0]
        self.best_model = models[best_name]
        self.best_model_name = best_name
        
        self.logger.info(f"Best model: {best_name}")
        
        return results
    
    def calibrate_models(self,
                        X_cal: np.ndarray,
                        y_cal: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate all trained models.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
        
        Returns:
            Dictionary of calibrated models
        """
        if not self.use_calibration:
            self.logger.info("Calibration disabled, skipping")
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info("CALIBRATING MODELS")
        self.logger.info("=" * 60)
        
        models = self.trainer.get_models()
        
        for name, model in models.items():
            self.calibrator.calibrate(model, X_cal, y_cal, model_name=name)
        
        return self.calibrator.get_calibrated_models()
    
    def create_ensemble(self,
                       X_val: np.ndarray,
                       y_val: np.ndarray) -> Any:
        """
        Create ensemble from trained models.
        
        Args:
            X_val: Validation features for weight calculation
            y_val: Validation labels
        
        Returns:
            Ensemble model
        """
        if not self.use_ensemble:
            self.logger.info("Ensemble disabled, using best single model")
            return self.best_model
        
        self.logger.info("=" * 60)
        self.logger.info("CREATING ENSEMBLE")
        self.logger.info("=" * 60)
        
        # Use calibrated models if available, otherwise use original
        if self.use_calibration:
            models = self.calibrator.get_calibrated_models()
        else:
            models = self.trainer.get_models()
        
        # Calculate optimal weights
        model_list, weights = self.ensemble_builder.create_weighted_average(
            models, X_val, y_val
        )
        
        # Create voting ensemble
        ensemble = self.ensemble_builder.create_voting_ensemble(
            model_list,
            voting='soft',
            weights=weights
        )
        
        # Fit the ensemble (required by sklearn VotingClassifier)
        # The individual models are already trained, this just sets up the ensemble
        self.logger.info("Fitting ensemble wrapper...")
        ensemble.fit(X_val, y_val)
        
        self.final_model = ensemble
        
        return ensemble
    
    def predict(self, X: np.ndarray, use_final: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            use_final: Use final model (ensemble) or best single model
        
        Returns:
            Predicted probabilities (n_samples, 9)
        """
        model = self.final_model if (use_final and self.final_model) else self.best_model
        
        if model is None:
            raise ValueError("No model available for prediction. Train models first.")
        
        return model.predict_proba(X)
    
    def save_model(self, filepath: str, model_type: str = 'final') -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
            model_type: 'final' for ensemble, 'best' for best single model
        """
        model = self.final_model if model_type == 'final' else self.best_model
        
        if model is None:
            self.logger.warning(f"No {model_type} model to save")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        self.logger.info(f"Saved {model_type} model to {filepath}")
    
    def load_model(self, filepath: str, model_type: str = 'final') -> Any:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
            model_type: 'final' or 'best'
        
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        if model_type == 'final':
            self.final_model = model
        else:
            self.best_model = model
        
        self.logger.info(f"Loaded {model_type} model from {filepath}")
        return model
    
    def get_summary(self) -> Dict:
        """
        Get summary of all training and evaluation results.
        
        Returns:
            Dictionary with complete summary
        """
        return {
            'cv_scores': self.trainer.get_cv_scores(),
            'evaluation_results': self.evaluator.get_results(),
            'best_model_name': self.best_model_name,
            'use_calibration': self.use_calibration,
            'use_ensemble': self.use_ensemble
        }
    
    def save_summary(self, filepath: str) -> None:
        """
        Save training summary to JSON file.
        
        Args:
            filepath: Path to save summary
        """
        summary = self.get_summary()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved summary to {filepath}")
