"""Ensemble models implementation."""

from typing import Dict, Any
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator

from .base import BaseModel

class ExtraTreesModel(BaseModel):
    """Extra Trees classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return ExtraTreesClassifier instance."""
        return ExtraTreesClassifier(**self.config)
        
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

class AdaBoostModel(BaseModel):
    """AdaBoost classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return AdaBoostClassifier instance."""
        return AdaBoostClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        return {
            'importance': self.model.feature_importances_,
            'estimator_weights': self.model.estimator_weights_,
            'n_estimators': self.model.n_estimators_
        }

class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return GradientBoostingClassifier instance."""
        return GradientBoostingClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        return {
            'importance': self.model.feature_importances_,
            'n_estimators': self.model.n_estimators_,
            'learning_rate': self.model.learning_rate
        } 