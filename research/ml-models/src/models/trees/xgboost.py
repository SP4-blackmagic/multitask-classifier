"""XGBoost classifier implementation."""

from typing import Dict
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
import numpy as np

from ..base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return XGBClassifier instance."""
        return XGBClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with different importance metrics
        """
        model = self.model
        
        # Get all importance types
        importance_scores = {
            'weight': model.get_score(importance_type='weight'),
            'gain': model.get_score(importance_type='gain'),
            'cover': model.get_score(importance_type='cover'),
            'total_gain': model.get_score(importance_type='total_gain'),
            'total_cover': model.get_score(importance_type='total_cover')
        }
        
        # Add feature contribution statistics
        feature_contribs = model.get_booster().predict(
            model.get_booster().DMatrix(np.zeros((1, model.n_features_in_))),
            pred_contribs=True
        )
        
        importance_scores.update({
            'feature_contributions': feature_contribs[0, :-1],  # Exclude bias term
            'bias': feature_contribs[0, -1]
        })
        
        # Add tree statistics
        trees_json = model.get_booster().get_dump(dump_format='json')
        importance_scores.update({
            'n_trees': len(trees_json),
            'avg_tree_weight': np.mean([float(tree['weight']) for tree in trees_json if 'weight' in tree])
        })
        
        return importance_scores 