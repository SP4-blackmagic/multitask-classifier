"""Linear Discriminant Analysis implementation."""

from typing import Dict, Any
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator

from ..base import BaseModel

class LDAModel(BaseModel):
    """Linear Discriminant Analysis implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return LinearDiscriminantAnalysis instance."""
        return LinearDiscriminantAnalysis(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        if not hasattr(self.model, 'coef_'):
            return {}
            
        return {
            'coefficients': self.model.coef_[0],
            'intercept': self.model.intercept_[0]
        } 