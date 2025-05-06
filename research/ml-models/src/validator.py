"""Validator module for model validation during training."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold
import logging

class ModelValidator:
    """Class for model validation during training."""
    
    def __init__(self, config: Dict):
        """Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_splits = config['training'].get('cv_folds', 5)
        self.random_state = config['training'].get('random_seed', 42)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def cross_validate(
        self,
        model: object,
        X: np.ndarray,
        y: np.ndarray,
        task: str
    ) -> Dict:
        """Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target labels
            task: Task name
            
        Returns:
            Dictionary containing validation metrics
        """
        try:
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                # Train model on this fold
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluate on validation set
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_val_fold)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = model.predict(X_val_fold)
                    
                # Calculate metrics
                accuracy = np.mean(y_pred == y_val_fold)
                fold_scores.append(accuracy)
                
                self.logger.info(f"Fold {fold}/{self.n_splits} - Accuracy: {accuracy:.4f}")
                
            # Calculate mean and std of scores
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            self.logger.info(
                f"Cross-validation results for {task} - "
                f"Mean accuracy: {mean_score:.4f} ± {std_score:.4f}"
            )
            
            return {
                'mean_accuracy': mean_score,
                'std_accuracy': std_score,
                'fold_scores': fold_scores
            }
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed for {task}: {e}")
            return {}
            
    def validate_hyperparameters(
        self,
        model_class: type,
        param_grid: Dict,
        X: np.ndarray,
        y: np.ndarray,
        task: str
    ) -> Optional[Dict]:
        """Validate hyperparameters using cross-validation.
        
        Args:
            model_class: Model class to validate
            param_grid: Grid of hyperparameters to try
            X: Feature matrix
            y: Target labels
            task: Task name
            
        Returns:
            Best hyperparameters or None if validation fails
        """
        try:
            from sklearn.model_selection import GridSearchCV
            
            grid_search = GridSearchCV(
                estimator=model_class(),
                param_grid=param_grid,
                cv=self.n_splits,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.logger.info(
                f"Best parameters for {task}: {grid_search.best_params_}\n"
                f"Best cross-validation score: {grid_search.best_score_:.4f}"
            )
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f"Hyperparameter validation failed for {task}: {e}")
            return None
