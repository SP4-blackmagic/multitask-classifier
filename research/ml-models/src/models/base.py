"""Base model class for all classifiers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import wandb
from sklearn.base import BaseEstimator

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self._model: Optional[BaseEstimator] = None
        self.model_name = self.__class__.__name__.replace('Model', '')
        
    @property
    def model(self) -> BaseEstimator:
        """Get the underlying sklearn model."""
        if self._model is None:
            self._model = self._create_model()
        return self._model
        
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create and return the actual model instance."""
        pass
        
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
        # Basic data checks
        checks = {
            f"{stage}/data_shape": X.shape,
            f"{stage}/n_samples": X.shape[0],
            f"{stage}/n_features": X.shape[1],
            f"{stage}/n_classes": len(np.unique(y)),
            f"{stage}/class_distribution": dict(zip(*np.unique(y, return_counts=True))),
            f"{stage}/missing_values": np.isnan(X).sum(),
            f"{stage}/infinite_values": np.isinf(X).sum(),
            f"{stage}/min_value": X.min(),
            f"{stage}/max_value": X.max(),
            f"{stage}/mean_value": X.mean(),
            f"{stage}/std_value": X.std()
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
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std()
                }
            })
            
    def log_model_info(self) -> None:
        """Log model information to wandb."""
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
        """
        # Log data sanity checks
        self.log_data_sanity_checks(X_train, y_train, 'train')
        if X_val is not None and y_val is not None:
            self.log_data_sanity_checks(X_val, y_val, 'val')
            
        # Log model information
        self.log_model_info()
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available.
        
        Returns:
            Feature importance scores or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        return None 