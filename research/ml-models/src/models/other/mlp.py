"""Multi-layer Perceptron classifier implementation."""

from typing import Dict, Any
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator

from ..base import BaseModel

class MLPModel(BaseModel):
    """Multi-layer Perceptron classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return MLPClassifier instance."""
        return MLPClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        if not hasattr(self.model, 'coefs_'):
            return {}
            
        # Calculate feature importance based on weights
        input_weights = self.model.coefs_[0]
        feature_importance = np.abs(input_weights).mean(axis=1)
        
        return {
            'weights': feature_importance,
            'n_layers': len(self.model.coefs_),
            'n_neurons': [coef.shape[1] for coef in self.model.coefs_]
        } 