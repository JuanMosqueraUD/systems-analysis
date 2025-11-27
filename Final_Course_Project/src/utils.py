"""
Utility functions for the Otto Group Classification System.

Simple helper functions for common operations like file handling,
logging setup, and basic validations.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def setup_logger(name: str = 'OttoClassifier', 
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with UTF-8 encoding for Windows compatibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # Force UTF-8 encoding to handle special characters
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_submission(predictions: np.ndarray, 
                    ids: np.ndarray,
                    output_path: str,
                    class_names: list = None) -> None:
    """
    Save predictions in Kaggle submission format.
    
    Args:
        predictions: Probability matrix (n_samples, 9)
        ids: Sample IDs
        output_path: Path to save CSV file
        class_names: List of class names (default: Class_1 to Class_9)
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(1, 10)]
    
    # Create submission dataframe
    submission_df = pd.DataFrame(predictions, columns=class_names)
    submission_df.insert(0, 'id', ids)
    
    # Ensure probabilities sum to 1
    prob_cols = [col for col in submission_df.columns if col.startswith('Class_')]
    submission_df[prob_cols] = submission_df[prob_cols].div(
        submission_df[prob_cols].sum(axis=1), axis=0
    )
    
    # Save
    ensure_dir(os.path.dirname(output_path))
    submission_df.to_csv(output_path, index=False)


def load_csv(file_path: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load CSV file with error handling.
    
    Args:
        file_path: Path to CSV file
        logger: Optional logger instance
    
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        if logger:
            logger.info(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        if logger:
            logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def calculate_logloss(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      eps: float = 1e-15) -> float:
    """
    Calculate multi-class logarithmic loss.
    
    Args:
        y_true: True class labels (0 to 8)
        y_pred_proba: Predicted probabilities (n_samples, 9)
        eps: Small value to clip probabilities
    
    Returns:
        Multi-class logloss value
    """
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Normalize to sum to 1
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Calculate log loss
    n_samples = y_true.shape[0]
    loss = -np.sum(np.log(y_pred_proba[np.arange(n_samples), y_true])) / n_samples
    
    return loss


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """Get the data directory path."""
    return get_project_root() / 'data'


def get_models_path() -> Path:
    """Get the models directory path."""
    return get_project_root() / 'models'


def get_outputs_path() -> Path:
    """Get the outputs directory path."""
    return get_project_root() / 'outputs'
