"""Quadratic Discriminant Analysis implementation."""

from typing import Dict, Any
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.base import BaseEstimator

from ..base import BaseModel

class QDAModel(BaseModel):
    """Quadratic Discriminant Analysis implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return QuadraticDiscriminantAnalysis instance."""
        return QuadraticDiscriminantAnalysis(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        # QDA doesn't provide direct feature importance
        return {} 