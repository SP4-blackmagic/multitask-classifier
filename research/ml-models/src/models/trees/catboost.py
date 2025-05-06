"""CatBoost classifier implementation."""

from typing import Dict
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
import numpy as np

from ..base import BaseModel

class CatBoostModel(BaseModel):
    """CatBoost classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return CatBoostClassifier instance."""
        return CatBoostClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with different importance metrics
        """
        model = self.model
        
        # Get all importance types
        importance_scores = {
            'PredictionValuesChange': model.get_feature_importance(),
            'LossFunctionChange': model.get_feature_importance(type='LossFunctionChange'),
            'ShapValues': model.get_feature_importance(type='ShapValues')
        }
        
        # Add model statistics
        tree_stats = model.get_tree_stats()
        if tree_stats:
            importance_scores.update({
                'n_trees': len(tree_stats),
                'avg_depth': np.mean([tree['depth'] for tree in tree_stats]),
                'avg_leaf_count': np.mean([tree['leaf_count'] for tree in tree_stats])
            })
            
        # Add interaction strengths if available
        try:
            interaction_strengths = model.get_feature_importance(
                type='Interaction',
                prettified=True
            )
            importance_scores['interaction_strengths'] = interaction_strengths
        except Exception:
            pass
            
        # Add per-feature statistics
        importance_scores.update({
            'per_feature_stats': model.get_feature_statistics()
        })
        
        return importance_scores 