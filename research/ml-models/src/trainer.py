"""Trainer module for model training."""

import os
import time
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import wandb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from .models.base import BaseModel
from .models.factory import ModelFactory
from .metrics import MetricsCalculator
from .validator import ModelValidator
from .visualizer import ModelVisualizer

class ModelTrainer:
    """Class for model training and evaluation."""
    
    def __init__(
        self,
        config: Dict,
        dataset_handler,
        output_dir: str,
        use_wandb: bool = True
    ):
        """Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            dataset_handler: Dataset handler instance
            output_dir: Directory to save models and results
            use_wandb: Whether to use wandb for logging
        """
        self.config = config
        self.dataset_handler = dataset_handler
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.metrics_calculator = MetricsCalculator()
        self.validator = ModelValidator(config)
        self.visualizer = ModelVisualizer(output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        task: str = 'ripeness'
    ) -> Tuple[BaseModel, Dict]:
        """Train a model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            task: Task name
            
        Returns:
            Tuple of (trained model, training history)
        """
        self.logger.info(f"Training {model_name} for {task}...")
        
        # Initialize wandb run if enabled
        if self.use_wandb:
            wandb.init(
                project="fruit-classification",
                name=f"{model_name}_{task}",
                config={
                    "model": model_name,
                    "task": task,
                    **self.config['model_registry'][model_name]
                }
            )
            
        # Create model using factory
        model = ModelFactory.create(model_name, self.config['model_registry'][model_name])
        
        # Train and time the model
        start_time = time.time()
        
        # Cross-validation
        cv_scores = self.validator.cross_validate(
            model.model,
            X_train,
            y_train,
            task
        )
        
        # Train on full training set with data sanity checks
        model.fit(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        # Calculate training metrics
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)
        train_metrics = self.metrics_calculator.calculate_metrics(
            y_train,
            y_train_pred,
            y_train_pred_proba
        )
        
        # Plot training visualizations
        self.visualizer.plot_confusion_matrix(
            train_metrics['confusion_matrix'],
            task,
            model_name,
            log_to_wandb=self.use_wandb
        )
        
        # Get and log feature importance if available
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            self.visualizer.plot_feature_importance(
                feature_importance,
                [f"feature_{i}" for i in range(X_train.shape[1])],
                task,
                model_name,
                log_to_wandb=self.use_wandb
            )
            
        # Calculate validation metrics if validation data is provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_pred_proba = model.predict_proba(X_val)
            val_metrics = self.metrics_calculator.calculate_metrics(
                y_val,
                y_val_pred,
                y_val_pred_proba
            )
            
            # Plot validation visualizations
            self.visualizer.plot_confusion_matrix(
                val_metrics['confusion_matrix'],
                task,
                model_name,
                log_to_wandb=self.use_wandb,
                prefix='val'
            )
            
            # Plot ROC and PR curves
            self.visualizer.plot_roc_curves(
                y_val,
                y_val_pred_proba,
                task,
                model_name,
                log_to_wandb=self.use_wandb
            )
            
            self.visualizer.plot_precision_recall_curves(
                y_val,
                y_val_pred_proba,
                task,
                model_name,
                log_to_wandb=self.use_wandb
            )
            
        # Log metrics to wandb
        if self.use_wandb:
            wandb.log({
                "training_time": training_time,
                "cv_mean_accuracy": cv_scores['mean_accuracy'],
                "cv_std_accuracy": cv_scores['std_accuracy'],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics else {})
            })
            
        # Save model
        model_path = os.path.join(self.output_dir, f"{model_name}_{task}.joblib")
        joblib.dump(model.model, model_path)
        
        # Log model to wandb
        if self.use_wandb:
            wandb.save(model_path)
            
        self.logger.info(f"Model saved to {model_path}")
        
        return model, {
            'training_time': training_time,
            'cv_scores': cv_scores,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
    def train_and_evaluate_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        task: str = 'ripeness'
    ) -> Dict:
        """Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            task: Task name
            
        Returns:
            Dictionary of results
        """
        results = {}
        model_names = list(self.config['model_registry'].keys())
        
        # Train and evaluate each model
        for model_name in tqdm(model_names, desc=f"Training models for {task}"):
            try:
                model, history = self.train_model(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    task
                )
                results[model_name] = history
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
                
        # Plot performance comparison
        if results:
            self.visualizer.plot_performance_comparison(
                'accuracy',
                pd.DataFrame([
                    {
                        'Task': task,
                        'Model': model_name,
                        'accuracy': history['val_metrics']['accuracy'] if history.get('val_metrics') else history['cv_scores']['mean_accuracy']
                    }
                    for model_name, history in results.items()
                ]),
                log_to_wandb=self.use_wandb
            )
            
        return results
        
    def create_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        task: str = 'ripeness',
        top_n: int = 3
    ) -> Tuple[StackingClassifier, Dict]:
        """Create an ensemble of models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            task: Task name
            top_n: Number of top models to include
            
        Returns:
            Tuple of (ensemble model, training history)
        """
        self.logger.info(f"Creating ensemble for {task}...")
        
        # Train all models
        results = self.train_and_evaluate_all_models(
            X_train,
            y_train,
            X_val,
            y_val,
            task
        )
        
        # Select top models based on validation accuracy
        if X_val is not None and y_val is not None:
            model_scores = {
                name: history['val_metrics']['accuracy']
                for name, history in results.items()
                if 'val_metrics' in history
            }
        else:
            model_scores = {
                name: history['cv_scores']['mean_accuracy']
                for name, history in results.items()
            }
            
        top_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Create base models
        base_models = []
        for model_name, _ in top_models:
            model = ModelFactory.create(model_name, self.config['model_registry'][model_name])
            base_models.append((model_name, model.model))
            
        # Create meta-model
        meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1
        )
        
        # Create and train ensemble
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=self.config['training']['n_cv_folds'],
            n_jobs=-1
        )
        
        start_time = time.time()
        
        # Cross-validation
        cv_scores = self.validator.cross_validate(
            ensemble,
            X_train,
            y_train,
            task
        )
        
        # Train on full training set
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Calculate training metrics
        y_train_pred = ensemble.predict(X_train)
        y_train_pred_proba = ensemble.predict_proba(X_train)
        train_metrics = self.metrics_calculator.calculate_metrics(
            y_train,
            y_train_pred,
            y_train_pred_proba
        )
        
        # Plot training visualizations
        self.visualizer.plot_confusion_matrix(
            train_metrics['confusion_matrix'],
            task,
            'ensemble',
            log_to_wandb=self.use_wandb
        )
        
        # Calculate validation metrics if validation data is provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            y_val_pred = ensemble.predict(X_val)
            y_val_pred_proba = ensemble.predict_proba(X_val)
            val_metrics = self.metrics_calculator.calculate_metrics(
                y_val,
                y_val_pred,
                y_val_pred_proba
            )
            
            # Plot validation visualizations
            self.visualizer.plot_confusion_matrix(
                val_metrics['confusion_matrix'],
                task,
                'ensemble',
                log_to_wandb=self.use_wandb,
                prefix='val'
            )
            
            # Plot ROC and PR curves
            self.visualizer.plot_roc_curves(
                y_val,
                y_val_pred_proba,
                task,
                'ensemble',
                log_to_wandb=self.use_wandb
            )
            
            self.visualizer.plot_precision_recall_curves(
                y_val,
                y_val_pred_proba,
                task,
                'ensemble',
                log_to_wandb=self.use_wandb
            )
            
        # Log metrics to wandb
        if self.use_wandb:
            wandb.log({
                "ensemble_training_time": training_time,
                "ensemble_cv_mean_accuracy": cv_scores['mean_accuracy'],
                "ensemble_cv_std_accuracy": cv_scores['std_accuracy'],
                **{f"ensemble_train_{k}": v for k, v in train_metrics.items()},
                **({f"ensemble_val_{k}": v for k, v in val_metrics.items()} if val_metrics else {})
            })
            
        # Save ensemble
        ensemble_path = os.path.join(self.output_dir, f"ensemble_{task}.joblib")
        joblib.dump(ensemble, ensemble_path)
        
        # Log ensemble to wandb
        if self.use_wandb:
            wandb.save(ensemble_path)
            
        self.logger.info(f"Ensemble saved to {ensemble_path}")
        
        return ensemble, {
            'training_time': training_time,
            'cv_scores': cv_scores,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'base_models': [name for name, _ in top_models]
        }
