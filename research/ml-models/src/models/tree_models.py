"""Tree-based models implementation."""

from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator

from .base import BaseModel

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
            
        # Get additional feature importance metrics
        return {
            'gini': base_importance,
            'permutation': self.model.feature_importances_,
            'n_nodes': self.model.n_features_in_
        }

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
        
        return {
            'weight': model.get_score(importance_type='weight'),
            'gain': model.get_score(importance_type='gain'),
            'cover': model.get_score(importance_type='cover'),
            'total_gain': model.get_score(importance_type='total_gain'),
            'total_cover': model.get_score(importance_type='total_cover')
        }

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
        
        return {
            'split': model.feature_importances_,
            'gain': model.booster_.feature_importance(importance_type='gain'),
            'split_count': model.booster_.feature_importance(importance_type='split')
        }

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
        
        return {
            'PredictionValuesChange': model.get_feature_importance(),
            'LossFunctionChange': model.get_feature_importance(type='LossFunctionChange'),
            'ShapValues': model.get_feature_importance(type='ShapValues')
        } 