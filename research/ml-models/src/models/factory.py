"""Model factory for creating model instances."""

from typing import Dict, Any, Optional, Type
from .base import BaseModel
from .tree_models import RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel
from .linear_models import LogisticRegressionModel, RidgeModel, SGDModel
from .svm_models import SVCLinearModel, SVCRBFModel
from .ensemble_models import ExtraTreesModel, AdaBoostModel, GradientBoostingModel
from .other import (
    KNNModel,
    DecisionTreeModel,
    GaussianNBModel,
    LDAModel,
    QDAModel,
    MLPModel
)

class ModelFactory:
    """Factory class for creating model instances."""
    
    _models: Dict[str, Type[BaseModel]] = {
        'RandomForest': RandomForestModel,
        'XGBoost': XGBoostModel,
        'LightGBM': LightGBMModel,
        'CatBoost': CatBoostModel,
        'LogisticRegression': LogisticRegressionModel,
        'Ridge': RidgeModel,
        'SGD': SGDModel,
        'SVC_linear': SVCLinearModel,
        'SVC_rbf': SVCRBFModel,
        'ExtraTrees': ExtraTreesModel,
        'AdaBoost': AdaBoostModel,
        'GradientBoosting': GradientBoostingModel,
        'KNN': KNNModel,
        'DecisionTree': DecisionTreeModel,
        'GaussianNB': GaussianNBModel,
        'LDA': LDAModel,
        'QDA': QDAModel,
        'MLP': MLPModel
    }
    
    @classmethod
    def create(cls, model_name: str, config: Dict[str, Any]) -> Optional[BaseModel]:
        """Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration dictionary
            
        Returns:
            Model instance or None if model_name is not found
            
        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_class = cls._models[model_name]
        return model_class(config)
        
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get dictionary of available models.
        
        Returns:
            Dictionary mapping model names to model classes
        """
        return cls._models.copy()
        
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model.
        
        Args:
            model_name: Name of the model
            model_class: Model class to register
        """
        cls._models[model_name] = model_class 