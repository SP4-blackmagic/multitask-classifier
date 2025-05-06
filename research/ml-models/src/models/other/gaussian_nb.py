"""Gaussian Naive Bayes classifier implementation."""

from typing import Dict, Any
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator

from ..base import BaseModel

class GaussianNBModel(BaseModel):
    """Gaussian Naive Bayes classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return GaussianNB instance."""
        return GaussianNB(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        # GaussianNB doesn't provide direct feature importance
        return {} 