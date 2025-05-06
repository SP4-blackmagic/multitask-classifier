"""Metrics module for model evaluation and reporting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

class ModelEvaluator:
    """Class for model evaluation and metrics calculation."""
    
    def __init__(self, output_dir: str):
        """Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.results = {}
        self.training_times = {}
        self.inference_times = {}
        
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: str,
        model_name: str,
        inference_time: float,
        training_time: Optional[float] = None,
        target_names: Optional[List[str]] = None,
        per_fruit_metrics: Optional[Dict] = None
    ) -> Dict:
        """Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            task: Task name
            model_name: Model name
            inference_time: Inference time per sample
            training_time: Training time (optional)
            target_names: Target class names (optional)
            per_fruit_metrics: Per-fruit metrics (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Store results
        if task not in self.results:
            self.results[task] = {}
            
        self.results[task][model_name] = {
            'overall': {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'inference_time_per_sample_ms': inference_time * 1000,
                'total_inference_time_s': inference_time * len(y_true)
            }
        }
        
        if training_time is not None:
            self.results[task][model_name]['overall']['training_time_s'] = training_time
            
        # Calculate classification report
        try:
            report = classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )
            self.results[task][model_name]['overall']['classification_report'] = report
        except Exception as e:
            print(f"Warning: Failed to generate classification report: {e}")
            
        # Store per-fruit metrics if provided
        if per_fruit_metrics:
            self.results[task][model_name]['per_fruit'] = per_fruit_metrics
            
        return self.results[task][model_name]
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: str,
        model_name: str,
        target_names: Optional[List[str]] = None,
        fruit_name: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            task: Task name
            model_name: Model name
            target_names: Target class names (optional)
            fruit_name: Fruit name for per-fruit plot (optional)
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
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
            
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to plot confusion matrix: {e}")
            
    def plot_performance_comparison(
        self,
        metric: str,
        results_df: pd.DataFrame
    ) -> None:
        """Plot performance comparison across models.
        
        Args:
            metric: Metric to plot
            results_df: DataFrame containing results
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
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to plot performance comparison: {e}")
            
    def save_results(self) -> None:
        """Save evaluation results to file."""
        try:
            results_to_save = {
                'test_metrics': self.results,
                'training_times': self.training_times,
                'inference_times': self.inference_times
            }
            
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
                    
            filename = f'test_set_benchmark_reports_{datetime.now().strftime("%Y%m%d")}.json'
            with open(os.path.join(self.output_dir, filename), 'w') as f:
                json.dump(results_to_save, f, indent=4, cls=NpEncoder)
                
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
            
    def generate_results_dataframe(self) -> pd.DataFrame:
        """Generate results DataFrame.
        
        Returns:
            DataFrame containing evaluation results
        """
        all_results_list = []
        
        for task, models_data in self.results.items():
            for model, metrics_dict in models_data.items():
                res = {'Task': task, 'Model': model}
                overall_metrics = metrics_dict.get('overall', {})
                
                res['Accuracy'] = overall_metrics.get('accuracy')
                res['Macro F1'] = overall_metrics.get('macro_f1')
                res['Weighted F1'] = overall_metrics.get('weighted_f1')
                res['Inference Time (ms/sample)'] = overall_metrics.get('inference_time_per_sample_ms')
                res['Training Time (s)'] = overall_metrics.get('training_time_s')
                
                all_results_list.append(res)
                
        if not all_results_list:
            return pd.DataFrame()
            
        results_df = pd.DataFrame(all_results_list)
        results_df = results_df.round(4)
        results_df = results_df.sort_values(by=['Task', 'Accuracy'], ascending=[True, False])
        
        return results_df
