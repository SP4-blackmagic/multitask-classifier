"""Decision Tree classifier implementation."""

from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

from ..base import BaseModel

class DecisionTreeModel(BaseModel):
    """Decision Tree classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return DecisionTreeClassifier instance."""
        return DecisionTreeClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        return {
            'gini': self.model.feature_importances_,
            'n_nodes': self.model.n_features_in_
        } 