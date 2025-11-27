"""
Data Processing Module for Otto Group Classification.

Handles data loading, validation, cleaning, and preprocessing.
Consolidates all data operations in a single, simple module.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Optional
import logging


class DataProcessor:
    """
    Unified data processing pipeline for Otto Group classification.
    
    Handles:
    - Data loading from CSV
    - Basic validation and cleaning
    - Feature scaling (StandardScaler)
    - Train/test splitting with stratification
    - Label encoding for target variable
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize DataProcessor.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'target'
        self.id_column = 'id'
        
    def load_train_data(self, file_path: str) -> pd.DataFrame:
        """
        Load training data from CSV.
        
        Args:
            file_path: Path to train.csv
        
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading training data from {file_path}")
        df = pd.read_csv(file_path)
        
        self.logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        self.logger.info(f"Target classes: {df[self.target_column].unique()}")
        self.logger.info(f"Class distribution:\n{df[self.target_column].value_counts()}")
        
        return df
    
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """
        Load test data from CSV.
        
        Args:
            file_path: Path to test.csv
        
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading test data from {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded {len(df)} test samples")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Perform basic validation on the dataset.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if validation passes
        """
        self.logger.info("Validating data...")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            self.logger.warning(f"Found {missing} missing values")
        else:
            self.logger.info("No missing values found")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
        
        # Check ID uniqueness
        if self.id_column in df.columns:
            if df[self.id_column].nunique() != len(df):
                self.logger.error("ID column contains duplicates!")
                return False
        
        self.logger.info("Data validation passed")
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset (handle missing values, outliers, etc.).
        
        Args:
            df: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Remove duplicates if any
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after = len(df_clean)
        if before != after:
            self.logger.info(f"Removed {before - after} duplicate rows")
        
        # Handle missing values (impute with median for numerical features)
        feature_cols = [col for col in df_clean.columns 
                       if col not in [self.id_column, self.target_column]]
        
        for col in feature_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        self.logger.info("Data cleaning completed")
        return df_clean
    
    def prepare_features_and_target(self, 
                                   df: pd.DataFrame,
                                   is_train: bool = True) -> Tuple:
        """
        Separate features, target, and ID from dataframe.
        
        Args:
            df: Input DataFrame
            is_train: Whether this is training data (has target)
        
        Returns:
            Tuple of (X, y, ids) or (X, ids) for test data
        """
        # Extract IDs
        if self.id_column in df.columns:
            ids = df[self.id_column].values
            df = df.drop(columns=[self.id_column])
        else:
            ids = np.arange(len(df))
        
        if is_train:
            # Extract target
            y = df[self.target_column].values
            X = df.drop(columns=[self.target_column])
            
            # Encode target labels (Class_1 -> 0, Class_2 -> 1, etc.)
            y_encoded = self.label_encoder.fit_transform(y)
            
            self.feature_columns = X.columns.tolist()
            self.logger.info(f"Prepared {len(self.feature_columns)} features")
            
            return X.values, y_encoded, ids
        else:
            # Test data - no target
            X = df
            if self.feature_columns:
                X = X[self.feature_columns]  # Ensure same column order
            return X.values, ids
    
    def scale_features(self, 
                      X_train: np.ndarray,
                      X_test: Optional[np.ndarray] = None,
                      fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply StandardScaler to features.
        
        Args:
            X_train: Training features
            X_test: Optional test features
            fit: Whether to fit the scaler (True for first call)
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if fit:
            self.logger.info("Fitting and transforming training data with StandardScaler")
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            self.logger.info("Transforming training data with fitted scaler")
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            self.logger.info("Transforming test data with fitted scaler")
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def create_train_val_split(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple:
        """
        Create stratified train/validation split.
        
        Args:
            X: Features
            y: Target labels
            test_size: Proportion for validation set
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        self.logger.info(f"Creating train/val split (test_size={test_size})")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Validation set: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    def get_cv_folds(self, 
                    n_splits: int = 5,
                    shuffle: bool = True,
                    random_state: int = 42) -> StratifiedKFold:
        """
        Create stratified K-fold cross-validation splitter.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
        
        Returns:
            StratifiedKFold splitter instance
        """
        self.logger.info(f"Creating {n_splits}-fold stratified CV")
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decode numerical labels back to original class names.
        
        Args:
            y_encoded: Encoded labels (0-8)
        
        Returns:
            Original class names (Class_1, Class_2, etc.)
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def process_train_pipeline(self, 
                              train_path: str,
                              validation_size: float = 0.2) -> Tuple:
        """
        Complete training data processing pipeline.
        
        Args:
            train_path: Path to train.csv
            validation_size: Proportion for validation set
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val, scaler, label_encoder)
        """
        # Load
        df_train = self.load_train_data(train_path)
        
        # Validate and clean
        self.validate_data(df_train)
        df_train = self.clean_data(df_train)
        
        # Prepare features and target
        X, y, _ = self.prepare_features_and_target(df_train, is_train=True)
        
        # Split
        X_train, X_val, y_train, y_val = self.create_train_val_split(
            X, y, test_size=validation_size
        )
        
        # Scale
        X_train, X_val = self.scale_features(X_train, X_val, fit=True)
        
        self.logger.info("Training pipeline completed successfully")
        return X_train, X_val, y_train, y_val
    
    def process_test_pipeline(self, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete test data processing pipeline.
        
        Args:
            test_path: Path to test.csv
        
        Returns:
            Tuple of (X_test_scaled, test_ids)
        """
        # Load
        df_test = self.load_test_data(test_path)
        
        # Validate and clean
        self.validate_data(df_test)
        df_test = self.clean_data(df_test)
        
        # Prepare features
        X_test, test_ids = self.prepare_features_and_target(df_test, is_train=False)
        
        # Scale (using already fitted scaler)
        _, X_test_scaled = self.scale_features(
            np.zeros((1, X_test.shape[1])),  # Dummy train data
            X_test,
            fit=False
        )
        
        self.logger.info("Test pipeline completed successfully")
        return X_test_scaled, test_ids
