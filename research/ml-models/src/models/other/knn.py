"""K-Nearest Neighbors classifier implementation."""

from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

from ..base import BaseModel

class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier implementation."""
    
    def _create_model(self) -> BaseEstimator:
        """Create and return KNeighborsClassifier instance."""
        return KNeighborsClassifier(**self.config)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance.
        
        Returns:
            Dictionary with feature importance metrics
        """
        # KNN doesn't provide direct feature importance
        return {} 