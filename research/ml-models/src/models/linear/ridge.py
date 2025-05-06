"""Ridge classifier implementation."""

from typing import Dict
from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator
import numpy as np

from ..base import BaseModel

class RidgeModel(BaseModel):
    """Ridge classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return RidgeClassifier instance."""
        return RidgeClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with different importance metrics
        """
        model = self.model
        
        # Get coefficients
        coef = model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_
        abs_coef = np.abs(coef)
        
        importance_scores = {
            'coefficients': coef,
            'abs_coefficients': abs_coef,
            'intercept': model.intercept_,
            'top_positive_features': np.argsort(coef)[-5:],
            'top_negative_features': np.argsort(coef)[:5],
            'mean_coefficient': np.mean(abs_coef),
            'std_coefficient': np.std(abs_coef),
            'l2_norm': np.sqrt(np.sum(coef ** 2)),
            'sparsity': np.mean(abs_coef < 1e-10)
        }
        
        # Add regularization strength
        if hasattr(model, 'alpha'):
            importance_scores['alpha'] = model.alpha
            
        return importance_scores 