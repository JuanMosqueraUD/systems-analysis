"""
Feature Engineering Module for Otto Group Classification.

Simple feature engineering with optional PCA and basic feature selection.
Keeps it minimal for university project scope.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from typing import Optional, Tuple
import logging


class FeatureEngineer:
    """
    Simple feature engineering pipeline.
    
    Provides:
    - PCA for dimensionality reduction (optional)
    - Low-variance feature removal
    - Basic feature statistics
    """
    
    def __init__(self, 
                 use_pca: bool = False,
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            use_pca: Whether to apply PCA
            n_components: Number of PCA components (None = keep all with variance)
            variance_threshold: Minimum variance for feature selection
            logger: Optional logger instance
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.pca = None
        self.variance_selector = None
        self.original_features = None
        
    def fit_transform(self, 
                     X_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None) -> Tuple:
        """
        Fit feature engineering pipeline and transform data.
        
        Args:
            X_train: Training features
            X_val: Optional validation features
        
        Returns:
            Tuple of (X_train_transformed, X_val_transformed)
        """
        self.original_features = X_train.shape[1]
        self.logger.info(f"Starting feature engineering with {self.original_features} features")
        
        X_train_transformed = X_train.copy()
        X_val_transformed = X_val.copy() if X_val is not None else None
        
        # 1. Remove low-variance features
        if self.variance_threshold > 0:
            X_train_transformed, X_val_transformed = self._apply_variance_threshold(
                X_train_transformed, X_val_transformed
            )
        
        # 2. Apply PCA if requested
        if self.use_pca:
            X_train_transformed, X_val_transformed = self._apply_pca(
                X_train_transformed, X_val_transformed
            )
        
        self.logger.info(f"Feature engineering completed: {X_train_transformed.shape[1]} features")
        return X_train_transformed, X_val_transformed
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted pipeline.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        """
        X_transformed = X.copy()
        
        if self.variance_selector is not None:
            X_transformed = self.variance_selector.transform(X_transformed)
        
        if self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)
        
        return X_transformed
    
    def _apply_variance_threshold(self,
                                  X_train: np.ndarray,
                                  X_val: Optional[np.ndarray]) -> Tuple:
        """
        Remove features with low variance.
        
        Args:
            X_train: Training features
            X_val: Optional validation features
        
        Returns:
            Tuple of filtered features
        """
        self.logger.info(f"Applying variance threshold: {self.variance_threshold}")
        
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_train_filtered = self.variance_selector.fit_transform(X_train)
        
        n_removed = X_train.shape[1] - X_train_filtered.shape[1]
        if n_removed > 0:
            self.logger.info(f"Removed {n_removed} low-variance features")
        
        X_val_filtered = None
        if X_val is not None:
            X_val_filtered = self.variance_selector.transform(X_val)
        
        return X_train_filtered, X_val_filtered
    
    def _apply_pca(self,
                  X_train: np.ndarray,
                  X_val: Optional[np.ndarray]) -> Tuple:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X_train: Training features
            X_val: Optional validation features
        
        Returns:
            Tuple of PCA-transformed features
        """
        self.logger.info(f"Applying PCA (n_components={self.n_components})")
        
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        self.logger.info(f"PCA: {X_train_pca.shape[1]} components explain "
                        f"{explained_var*100:.2f}% variance")
        
        X_val_pca = None
        if X_val is not None:
            X_val_pca = self.pca.transform(X_val)
        
        return X_train_pca, X_val_pca
    
    def get_pca_info(self) -> dict:
        """
        Get information about PCA transformation.
        
        Returns:
            Dictionary with PCA statistics
        """
        if self.pca is None:
            return {}
        
        return {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            'total_variance_explained': self.pca.explained_variance_ratio_.sum()
        }
    
    def get_feature_info(self) -> dict:
        """
        Get information about feature engineering results.
        
        Returns:
            Dictionary with feature statistics
        """
        info = {
            'original_features': self.original_features,
            'variance_threshold': self.variance_threshold,
            'use_pca': self.use_pca
        }
        
        if self.variance_selector is not None:
            info['features_after_variance_filter'] = \
                self.variance_selector.get_support().sum()
        
        if self.pca is not None:
            info.update(self.get_pca_info())
        
        return info


def create_simple_feature_engineer(logger: Optional[logging.Logger] = None) -> FeatureEngineer:
    """
    Create a basic feature engineer (no PCA, no variance filtering).
    
    Args:
        logger: Optional logger
    
    Returns:
        FeatureEngineer instance
    """
    return FeatureEngineer(
        use_pca=False,
        variance_threshold=0.0,
        logger=logger
    )


def create_pca_feature_engineer(n_components: int = 50,
                               logger: Optional[logging.Logger] = None) -> FeatureEngineer:
    """
    Create a feature engineer with PCA dimensionality reduction.
    
    Args:
        n_components: Number of PCA components to keep
        logger: Optional logger
    
    Returns:
        FeatureEngineer instance configured with PCA
    """
    return FeatureEngineer(
        use_pca=True,
        n_components=n_components,
        variance_threshold=0.01,  # Small threshold to remove near-zero variance
        logger=logger
    )
