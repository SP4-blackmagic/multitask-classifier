"""Metrics module for calculating evaluation metrics."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

class MetricsCalculator:
    """Class for calculating evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        target_names: Optional[List[str]] = None
    ) -> Dict:
        """Calculate all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            target_names: Target class names (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except Exception:
                metrics['roc_auc'] = None
                metrics['average_precision'] = None
                
        return metrics
        
    @staticmethod
    def calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        target_names: Optional[List[str]] = None
    ) -> Dict:
        """Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            target_names: Target class names (optional)
            
        Returns:
            Dictionary containing per-class metrics
        """
        per_class_metrics = {}
        
        # Get unique classes
        classes = np.unique(y_true)
        if target_names is None:
            target_names = [f'class_{i}' for i in classes]
            
        # Calculate metrics for each class
        for i, class_name in enumerate(target_names):
            class_metrics = {}
            
            # Binary metrics for this class
            y_true_binary = (y_true == classes[i])
            y_pred_binary = (y_pred == classes[i])
            
            class_metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            class_metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            class_metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Probability-based metrics
            if y_pred_proba is not None:
                try:
                    class_metrics['roc_auc'] = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                    class_metrics['average_precision'] = average_precision_score(y_true_binary, y_pred_proba[:, i])
                except Exception:
                    class_metrics['roc_auc'] = None
                    class_metrics['average_precision'] = None
                    
            per_class_metrics[class_name] = class_metrics
            
        return per_class_metrics
        
    @staticmethod
    def calculate_per_fruit_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fruit_ids: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        target_names: Optional[List[str]] = None
    ) -> Dict:
        """Calculate metrics for each fruit type.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            fruit_ids: Fruit IDs
            y_pred_proba: Predicted probabilities (optional)
            target_names: Target class names (optional)
            
        Returns:
            Dictionary containing per-fruit metrics
        """
        per_fruit_metrics = {}
        
        # Get unique fruit IDs
        unique_fruits = np.unique(fruit_ids)
        
        # Calculate metrics for each fruit
        for fruit_id in unique_fruits:
            # Get indices for this fruit
            fruit_mask = (fruit_ids == fruit_id)
            
            # Calculate metrics
            fruit_metrics = MetricsCalculator.calculate_metrics(
                y_true[fruit_mask],
                y_pred[fruit_mask],
                y_pred_proba[fruit_mask] if y_pred_proba is not None else None,
                target_names
            )
            
            per_fruit_metrics[fruit_id] = fruit_metrics
            
        return per_fruit_metrics
