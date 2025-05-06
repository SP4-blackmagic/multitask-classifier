"""Random Forest classifier implementation."""

from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from ..base import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return RandomForestClassifier instance."""
        return RandomForestClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with additional importance metrics
        """
        base_importance = super().get_feature_importance()
        if base_importance is None:
            return {}
            
        return {
            'gini': base_importance,
            'permutation': self.model.feature_importances_,
            'n_nodes': self.model.n_features_in_,
            'n_estimators': len(self.model.estimators_),
            'max_depth': max(tree.tree_.max_depth for tree in self.model.estimators_),
            'min_depth': min(tree.tree_.max_depth for tree in self.model.estimators_),
            'avg_depth': sum(tree.tree_.max_depth for tree in self.model.estimators_) / len(self.model.estimators_)
        } 