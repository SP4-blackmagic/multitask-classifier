"""LightGBM classifier implementation."""

from typing import Dict
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
import numpy as np

from ..base import BaseModel

class LightGBMModel(BaseModel):
    """LightGBM classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return LGBMClassifier instance."""
        return LGBMClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with different importance metrics
        """
        model = self.model
        booster = model.booster_
        
        # Get all importance types
        importance_scores = {
            'split': model.feature_importances_,
            'gain': booster.feature_importance(importance_type='gain'),
            'split_count': booster.feature_importance(importance_type='split')
        }
        
        # Add model statistics
        importance_scores.update({
            'n_trees': booster.num_trees(),
            'n_features': booster.num_feature(),
            'max_depth': max(tree.max_depth for tree in booster.dump_model()['tree_info']),
            'avg_leaf_count': np.mean([tree.num_leaves for tree in booster.dump_model()['tree_info']])
        })
        
        # Add SHAP values if available
        try:
            shap_values = booster.predict(model._Booster__data, pred_contrib=True)
            importance_scores.update({
                'shap_values': np.abs(shap_values).mean(axis=0),
                'shap_interaction': np.abs(shap_values).sum(axis=1).mean()
            })
        except Exception:
            pass
            
        return importance_scores 