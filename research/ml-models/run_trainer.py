#!/usr/bin/env python3
"""CLI script for training models."""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.trainer import ModelTrainer
from src.dataset import SpectralDataset

def load_config(config_path: str) -> Dict:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required sections are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Load base config if specified
    if 'base_config' in config:
        base_config_path = os.path.join(os.path.dirname(config_path), config['base_config'])
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")
            
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
            
        # Merge configs, with model-specific config taking precedence
        config = {**base_config, **config}
        
    # Load data config if specified
    if 'data_config' in config:
        data_config_path = os.path.join(os.path.dirname(config_path), config['data_config'])
        if not os.path.exists(data_config_path):
            raise FileNotFoundError(f"Data configuration file not found: {data_config_path}")
            
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Add data config to the merged config
        config['data'] = data_config
        
    # Validate required configuration sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
            
    return config

def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description='Train machine learning models')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save models and results'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset handler
        logger.info("Initializing dataset handler")
        dataset_handler = SpectralDataset(config['data'])
        
        # Initialize trainer
        logger.info("Initializing model trainer")
        trainer = ModelTrainer(
            config=config,
            dataset_handler=dataset_handler,
            output_dir=str(output_dir),
            use_wandb=not args.no_wandb
        )
        
        # Get training data
        logger.info("Loading training data")
        X_train, y_train = dataset_handler.get_training_data()
        X_val, y_val = dataset_handler.get_validation_data()
        
        # Train models for each task
        for task in config['training']['tasks']:
            logger.info(f"Training models for task: {task}")
            
            # Get task-specific data
            X_train_task, y_train_task = dataset_handler.get_task_data(X_train, y_train, task)
            X_val_task, y_val_task = dataset_handler.get_task_data(X_val, y_val, task)
            
            # Train each model specified in the config
            for model_name in config['training']['models']:
                logger.info(f"Training {model_name} for {task}")
                model, history = trainer.train_model(
                    model_name=model_name,
                    X_train=X_train_task,
                    y_train=y_train_task,
                    X_val=X_val_task,
                    y_val=y_val_task,
                    task=task
                )
                
                logger.info(f"Training completed for {model_name} on {task}")
                logger.info(f"Training metrics: {history['train_metrics']}")
                if history['val_metrics']:
                    logger.info(f"Validation metrics: {history['val_metrics']}")
                    
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
