"""Visualizer module for plotting and visualization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import wandb

class ModelVisualizer:
    """Class for model visualization and plotting."""
    
    def __init__(self, output_dir: str):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        task: str,
        model_name: str,
        target_names: Optional[List[str]] = None,
        fruit_name: Optional[str] = None,
        log_to_wandb: bool = True
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            task: Task name
            model_name: Model name
            target_names: Target class names (optional)
            fruit_name: Fruit name for per-fruit plot (optional)
            log_to_wandb: Whether to log to wandb
        """
        try:
            plt.figure(figsize=(max(5, len(target_names)), max(4, len(target_names) * 0.8)))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names
            )
            
            title = f'Overall Test CM - {task.capitalize()} - {model_name}'
            if fruit_name:
                title = f'Test CM - {task.capitalize()} - {model_name} ({fruit_name})'
                
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(title)
            
            # Save plot
            filename = f'confusion_matrix_{task}_{model_name}'
            if fruit_name:
                filename += f'_fruit_{fruit_name}'
            filename += f'_{datetime.now().strftime("%Y%m%d")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f"{task}/{model_name}/confusion_matrix": wandb.Image(filepath)
                })
                
        except Exception as e:
            print(f"Warning: Failed to plot confusion matrix: {e}")
            
    def plot_performance_comparison(
        self,
        metric: str,
        results_df: pd.DataFrame,
        log_to_wandb: bool = True
    ) -> None:
        """Plot performance comparison across models.
        
        Args:
            metric: Metric to plot
            results_df: DataFrame containing results
            log_to_wandb: Whether to log to wandb
        """
        try:
            df_plot = results_df[['Task', 'Model', metric]].dropna().copy()
            df_plot['Task'] = df_plot['Task'].str.capitalize()
            df_plot = df_plot.sort_values(by=['Task', metric], ascending=[True, False])
            
            if df_plot.empty:
                return
                
            num_tasks = df_plot['Task'].nunique()
            tasks_plotted = df_plot['Task'].unique()
            num_models_per_task = df_plot.groupby('Task')['Model'].nunique().max()
            fig_height_per_task = max(4, 0.4 * num_models_per_task)
            total_fig_height = fig_height_per_task * num_tasks
            
            fig, axes = plt.subplots(
                num_tasks,
                1,
                figsize=(12, total_fig_height),
                squeeze=False
            )
            
            fig.suptitle(
                f'Model Comparison: Overall Test Set {metric}',
                fontsize=16,
                y=1.01
            )
            
            for i, task_name in enumerate(tasks_plotted):
                ax = axes[i, 0]
                task_data = df_plot[df_plot['Task'] == task_name]
                
                sns.barplot(
                    x=metric,
                    y='Model',
                    data=task_data,
                    ax=ax,
                    palette='viridis',
                    orient='h'
                )
                
                ax.bar_label(ax.containers[0], fmt='%.3f', padding=3)
                ax.set_title(f'Task: {task_name}')
                ax.set_xlabel(metric)
                ax.set_ylabel('Model')
                ax.set_xlim(0, max(1.05, task_data[metric].max() * 1.1))
                ax.grid(True, axis='x', linestyle='--', alpha=0.6)
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            
            # Save plot
            filename = f'model_{metric.lower().replace(" ", "_")}_comparison_{datetime.now().strftime("%Y%m%d")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f"model_comparison/{metric}": wandb.Image(filepath)
                })
                
        except Exception as e:
            print(f"Warning: Failed to plot performance comparison: {e}")
            
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        task: str,
        model_name: str,
        metric: str = 'Accuracy',
        log_to_wandb: bool = True
    ) -> None:
        """Plot learning curves.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            task: Task name
            model_name: Model name
            metric: Metric name
            log_to_wandb: Whether to log to wandb
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_scores, label='Training')
            plt.plot(val_scores, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'Learning Curves - {task} - {model_name}')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            filename = f'learning_curves_{task}_{model_name}_{datetime.now().strftime("%Y%m%d")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f"{task}/{model_name}/learning_curves": wandb.Image(filepath)
                })
                
        except Exception as e:
            print(f"Warning: Failed to plot learning curves: {e}")
            
    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: List[str],
        task: str,
        model_name: str,
        top_n: int = 20,
        log_to_wandb: bool = True
    ) -> None:
        """Plot feature importance.
        
        Args:
            feature_importance: Feature importance scores
            feature_names: Feature names
            task: Task name
            model_name: Model name
            top_n: Number of top features to plot
            log_to_wandb: Whether to log to wandb
        """
        try:
            # Sort features by importance
            indices = np.argsort(feature_importance)[-top_n:]
            plt.figure(figsize=(10, max(6, top_n * 0.4)))
            plt.barh(range(top_n), feature_importance[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Features - {task} - {model_name}')
            
            # Save plot
            filename = f'feature_importance_{task}_{model_name}_{datetime.now().strftime("%Y%m%d")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f"{task}/{model_name}/feature_importance": wandb.Image(filepath)
                })
                
        except Exception as e:
            print(f"Warning: Failed to plot feature importance: {e}")
            
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        task: str,
        model_name: str,
        target_names: Optional[List[str]] = None,
        log_to_wandb: bool = True
    ) -> None:
        """Plot ROC curves.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            task: Task name
            model_name: Model name
            target_names: Target class names (optional)
            log_to_wandb: Whether to log to wandb
        """
        try:
            from sklearn.metrics import roc_curve, auc
            
            plt.figure(figsize=(10, 8))
            
            # Get unique classes
            classes = np.unique(y_true)
            if target_names is None:
                target_names = [f'Class {i}' for i in classes]
                
            # Plot ROC curve for each class
            for i, class_name in enumerate(target_names):
                y_true_binary = (y_true == classes[i])
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(
                    fpr,
                    tpr,
                    label=f'{class_name} (AUC = {roc_auc:.3f})'
                )
                
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {task} - {model_name}')
            plt.legend(loc="lower right")
            
            # Save plot
            filename = f'roc_curves_{task}_{model_name}_{datetime.now().strftime("%Y%m%d")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f"{task}/{model_name}/roc_curves": wandb.Image(filepath)
                })
                
        except Exception as e:
            print(f"Warning: Failed to plot ROC curves: {e}")
            
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        task: str,
        model_name: str,
        target_names: Optional[List[str]] = None,
        log_to_wandb: bool = True
    ) -> None:
        """Plot precision-recall curves.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            task: Task name
            model_name: Model name
            target_names: Target class names (optional)
            log_to_wandb: Whether to log to wandb
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            plt.figure(figsize=(10, 8))
            
            # Get unique classes
            classes = np.unique(y_true)
            if target_names is None:
                target_names = [f'Class {i}' for i in classes]
                
            # Plot PR curve for each class
            for i, class_name in enumerate(target_names):
                y_true_binary = (y_true == classes[i])
                precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
                avg_precision = average_precision_score(y_true_binary, y_pred_proba[:, i])
                
                plt.plot(
                    recall,
                    precision,
                    label=f'{class_name} (AP = {avg_precision:.3f})'
                )
                
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves - {task} - {model_name}')
            plt.legend(loc="lower left")
            
            # Save plot
            filename = f'pr_curves_{task}_{model_name}_{datetime.now().strftime("%Y%m%d")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    f"{task}/{model_name}/pr_curves": wandb.Image(filepath)
                })
                
        except Exception as e:
            print(f"Warning: Failed to plot precision-recall curves: {e}") 