"""Logistic Regression classifier implementation."""

from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import numpy as np

from ..base import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return LogisticRegression instance."""
        return LogisticRegression(**self.config)
        
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
            'sparsity': np.mean(abs_coef < 1e-10)
        }
        
        # Add L1/L2 regularization metrics if available
        if hasattr(model, 'l1_ratio_'):
            importance_scores['l1_ratio'] = model.l1_ratio_
            
        return importance_scores 