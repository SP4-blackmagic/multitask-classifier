"""Tester module for model testing and inference."""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import joblib
import logging
from .metrics import ModelEvaluator

class ModelTester:
    """Class for model testing and inference."""
    
    def __init__(self, config: Dict, output_dir: str):
        """Initialize the tester.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save test results
        """
        self.config = config
        self.output_dir = output_dir
        self.evaluator = ModelEvaluator(output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path: str) -> Optional[object]:
        """Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None
                
            model = joblib.load(model_path)
            self.logger.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return None
            
    def test_model(
        self,
        model: object,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        model_name: str,
        target_names: Optional[List[str]] = None,
        per_fruit_metrics: Optional[Dict] = None
    ) -> Dict:
        """Test a model on test data.
        
        Args:
            model: Trained model
            X: Test feature matrix
            y: Test labels
            task: Task name
            model_name: Model name
            target_names: Target class names (optional)
            per_fruit_metrics: Per-fruit metrics (optional)
            
        Returns:
            Dictionary containing test metrics
        """
        try:
            # Measure inference time
            start_time = time.time()
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X)
            inference_time = (time.time() - start_time) / len(X)
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(
                y_true=y,
                y_pred=y_pred,
                task=task,
                model_name=model_name,
                inference_time=inference_time,
                target_names=target_names,
                per_fruit_metrics=per_fruit_metrics
            )
            
            # Plot confusion matrix
            self.evaluator.plot_confusion_matrix(
                y_true=y,
                y_pred=y_pred,
                task=task,
                model_name=model_name,
                target_names=target_names
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to test {model_name} for {task}: {e}")
            return {}
            
    def test_all_models(
        self,
        models_dir: str,
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray],
        target_names: Dict[str, List[str]]
    ) -> None:
        """Test all saved models.
        
        Args:
            models_dir: Directory containing saved models
            X_test: Test feature matrix
            y_test: Dictionary of test labels for each task
            target_names: Dictionary of target class names for each task
        """
        for task, y_test_task in y_test.items():
            self.logger.info(f"Testing models for task: {task}")
            
            # Find all model files for this task
            model_files = [
                f for f in os.listdir(models_dir)
                if f.endswith(f'_model_{task}.joblib')
            ]
            
            for model_file in model_files:
                model_name = model_file.replace(f'_model_{task}.joblib', '')
                model_path = os.path.join(models_dir, model_file)
                
                # Load and test model
                model = self.load_model(model_path)
                if model is not None:
                    self.test_model(
                        model=model,
                        X=X_test,
                        y=y_test_task,
                        task=task,
                        model_name=model_name,
                        target_names=target_names.get(task)
                    )
                    
        # Generate and save results
        results_df = self.evaluator.generate_results_dataframe()
        self.evaluator.save_results()
        
        # Plot performance comparisons
        for metric in ['Accuracy', 'Macro F1']:
            self.evaluator.plot_performance_comparison(metric, results_df)
            
    def predict(
        self,
        model: object,
        X: np.ndarray,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using a model.
        
        Args:
            model: Trained model
            X: Feature matrix
            return_proba: Whether to return probability estimates
            
        Returns:
            Predicted labels and optionally probability estimates
        """
        try:
            if return_proba and hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
                return y_pred, y_pred_proba
            else:
                y_pred = model.predict(X)
                return y_pred
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None 