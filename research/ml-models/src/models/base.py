"""Base model class for all classifiers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import wandb
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model.
        
        Args:
            config: Model configuration dictionary
            
        Raises:
            ValueError: If config is empty or invalid
        """
        if not config:
            raise ValueError("Configuration dictionary cannot be empty")
            
        self.config = config
        self._model: Optional[BaseEstimator] = None
        self.model_name = self.__class__.__name__.replace('Model', '')
        self._is_fitted = False
        
    @property
    def model(self) -> BaseEstimator:
        """Get the underlying sklearn model.
        
        Returns:
            The sklearn model instance
            
        Raises:
            RuntimeError: If model creation fails
        """
        if self._model is None:
            try:
                self._model = self._create_model()
            except Exception as e:
                raise RuntimeError(f"Failed to create model: {str(e)}")
        return self._model
        
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create and return the actual model instance.
        
        Returns:
            A sklearn model instance
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _create_model")
        
    def validate_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stage: str = 'train'
    ) -> None:
        """Validate input data.
        
        Args:
            X: Feature matrix
            y: Target labels
            stage: Data stage (train/val/test)
            
        Raises:
            ValueError: If data validation fails
        """
        if X is None or y is None:
            raise ValueError(f"{stage} data cannot be None")
            
        if len(X) != len(y):
            raise ValueError(f"{stage} X and y must have the same length")
            
        if X.size == 0 or y.size == 0:
            raise ValueError(f"{stage} data cannot be empty")
            
        if np.isnan(X).any():
            raise ValueError(f"{stage} data contains NaN values")
            
        if np.isinf(X).any():
            raise ValueError(f"{stage} data contains infinite values")
            
    def log_data_sanity_checks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stage: str = 'train'
    ) -> None:
        """Log data sanity checks to wandb.
        
        Args:
            X: Feature matrix
            y: Target labels
            stage: Data stage (train/val/test)
        """
        try:
            # Validate data first
            self.validate_data(X, y, stage)
            
            # Basic data checks
            checks = {
                f"{stage}/data_shape": X.shape,
                f"{stage}/n_samples": X.shape[0],
                f"{stage}/n_features": X.shape[1],
                f"{stage}/n_classes": len(np.unique(y)),
                f"{stage}/class_distribution": dict(zip(*np.unique(y, return_counts=True))),
                f"{stage}/missing_values": np.isnan(X).sum(),
                f"{stage}/infinite_values": np.isinf(X).sum(),
                f"{stage}/min_value": float(X.min()),
                f"{stage}/max_value": float(X.max()),
                f"{stage}/mean_value": float(X.mean()),
                f"{stage}/std_value": float(X.std())
            }
            
            # Feature statistics
            feature_stats = {
                'min': X.min(axis=0),
                'max': X.max(axis=0),
                'mean': X.mean(axis=0),
                'std': X.std(axis=0),
                'zeros': (X == 0).sum(axis=0) / X.shape[0],
                'missing': np.isnan(X).sum(axis=0) / X.shape[0]
            }
            
            # Log basic checks
            wandb.log(checks)
            
            # Log feature statistics as plots
            for stat_name, values in feature_stats.items():
                wandb.log({
                    f"{stage}/feature_{stat_name}_distribution": wandb.Histogram(values),
                    f"{stage}/feature_{stat_name}_stats": {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
                })
        except Exception as e:
            print(f"Warning: Failed to log data sanity checks: {str(e)}")
            
    def log_model_info(self) -> None:
        """Log model information to wandb."""
        try:
            # Get model parameters
            params = self.model.get_params()
            
            # Log model architecture info
            model_info = {
                'model_name': self.model_name,
                'model_class': self.model.__class__.__name__,
                'n_parameters': len(params),
                'parameters': params
            }
            
            wandb.log({'model_info': model_info})
        except Exception as e:
            print(f"Warning: Failed to log model info: {str(e)}")
            
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit the model with data sanity checks.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Raises:
            ValueError: If data validation fails
            RuntimeError: If model fitting fails
        """
        try:
            # Validate and log data
            self.validate_data(X_train, y_train, 'train')
            self.log_data_sanity_checks(X_train, y_train, 'train')
            
            if X_val is not None and y_val is not None:
                self.validate_data(X_val, y_val, 'val')
                self.log_data_sanity_checks(X_val, y_val, 'val')
                
            # Log model information
            self.log_model_info()
            
            # Fit the model
            self.model.fit(X_train, y_train)
            self._is_fitted = True
            
        except Exception as e:
            raise RuntimeError(f"Model fitting failed: {str(e)}")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
            
        Raises:
            NotFittedError: If model is not fitted
            ValueError: If input data is invalid
        """
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before making predictions")
            
        self.validate_data(X, np.zeros(len(X)), 'predict')
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
            
        Raises:
            NotFittedError: If model is not fitted
            ValueError: If input data is invalid
            AttributeError: If model doesn't support predict_proba
        """
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before making predictions")
            
        self.validate_data(X, np.zeros(len(X)), 'predict_proba')
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"{self.model_name} does not support probability predictions")
            
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available.
        
        Returns:
            Feature importance scores or None
            
        Raises:
            NotFittedError: If model is not fitted
        """
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before getting feature importance")
            
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        return None 