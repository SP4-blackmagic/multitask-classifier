"""SVM models implementation."""

from typing import Dict, Any
from sklearn.svm import SVC, LinearSVC
from sklearn.base import BaseEstimator

from .base import BaseModel

class SVCLinearModel(BaseModel):
    """Linear SVC implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return SVC with linear kernel instance."""
        return SVC(kernel='linear', **self.config)
        
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

class SVCRBFModel(BaseModel):
    """RBF SVC implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return SVC with RBF kernel instance."""
        return SVC(kernel='rbf', **self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        # RBF kernel doesn't provide direct feature importance
        # We can use permutation importance or SHAP values
        return {} 