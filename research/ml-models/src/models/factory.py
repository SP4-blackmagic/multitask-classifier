"""Model factory for creating model instances."""

from typing import Dict, Any, Optional, Type, Set
from .base import BaseModel
from .tree_models import RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel
from .linear.logistic_regression import LogisticRegressionModel
from .linear.ridge import RidgeModel
from .linear.sgd import SGDModel
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
    
    # Define model categories for better organization
    TREE_MODELS: Set[str] = {'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'DecisionTree'}
    LINEAR_MODELS: Set[str] = {'LogisticRegression', 'Ridge', 'SGD'}
    SVM_MODELS: Set[str] = {'SVC_linear', 'SVC_rbf'}
    ENSEMBLE_MODELS: Set[str] = {'ExtraTrees', 'AdaBoost', 'GradientBoosting'}
    OTHER_MODELS: Set[str] = {'KNN', 'GaussianNB', 'LDA', 'QDA', 'MLP'}
    
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
    def create(cls, model_name: str, config: Dict[str, Any]) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_name is not recognized or config is invalid
            RuntimeError: If model creation fails
        """
        if not model_name:
            raise ValueError("Model name cannot be empty")
            
        if not config:
            raise ValueError("Configuration dictionary cannot be empty")
            
        if model_name not in cls._models:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {', '.join(sorted(cls._models.keys()))}"
            )
            
        try:
            model_class = cls._models[model_name]
            return model_class(config)
        except Exception as e:
            raise RuntimeError(f"Failed to create model {model_name}: {str(e)}")
            
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get dictionary of available models.
        
        Returns:
            Dictionary mapping model names to model classes
        """
        return cls._models.copy()
        
    @classmethod
    def get_models_by_category(cls) -> Dict[str, Set[str]]:
        """Get models grouped by category.
        
        Returns:
            Dictionary mapping categories to sets of model names
        """
        return {
            'Tree Models': cls.TREE_MODELS,
            'Linear Models': cls.LINEAR_MODELS,
            'SVM Models': cls.SVM_MODELS,
            'Ensemble Models': cls.ENSEMBLE_MODELS,
            'Other Models': cls.OTHER_MODELS
        }
        
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model.
        
        Args:
            model_name: Name of the model
            model_class: Model class to register
            
        Raises:
            ValueError: If model_name is empty or model_class is invalid
        """
        if not model_name:
            raise ValueError("Model name cannot be empty")
            
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel, got {model_class}")
            
        cls._models[model_name] = model_class
        
    @classmethod
    def unregister_model(cls, model_name: str) -> None:
        """Unregister a model.
        
        Args:
            model_name: Name of the model to unregister
            
        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in cls._models:
            raise ValueError(f"Model {model_name} is not registered")
            
        del cls._models[model_name] 