import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import logging
import os

@dataclass
class DatasetContainer:
    """Standardized return contract for the data engine."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    task_type: str  # 'binary', 'multiclass', 'regression'
    feature_names: List[str]
    target_name: str
    metadata: dict

class DataLoader:
    """Handles CSV ingestion, task inference, and reproducible splitting."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DataLoader")

    def infer_task(self, y: pd.Series) -> str:
        """Infers the machine learning task based on target values."""
        if pd.api.types.is_numeric_dtype(y):
            # Check if it's integer-like with few unique values
            unique_vals = y.dropna().unique()
            if len(unique_vals) <= 2:
                return "binary"
            if len(unique_vals) <= 10 and all(float(val).is_integer() for val in unique_vals):
                return "multiclass"
            return "regression"
        else:
            unique_vals = y.dropna().unique()
            return "binary" if len(unique_vals) == 2 else "multiclass"

    def load_local_csv(
        self, 
        file_path: str, 
        target_column: Optional[str] = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        task_override: Optional[str] = None
    ) -> DatasetContainer:
        """Loads a CSV, infers task, and splits into train/val/test."""
        
        self.logger.info(f"Loading dataset from: {file_path}")
        
        # 1. Ingestion (auto-delimiter detection)
        try:
            df = pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            raise

        # 2. Target Resolution
        if target_column is None:
            target_column = df.columns[-1]
            self.logger.info(f"Auto-selected last column as target: {target_column}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 3. Task Inference
        task_type = task_override if task_override else self.infer_task(y)
        self.logger.info(f"Inferred task type: {task_type}")

        # 4. Reproducible Split
        # First split off the test set
        stratify = y if task_type in ["binary", "multiclass"] else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=stratify
        )
        
        # Then split remaining into train and val
        # Adjust val_size to be relative to the temp set
        relative_val_size = val_size / (1 - test_size)
        stratify_temp = y_temp if task_type in ["binary", "multiclass"] else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=self.seed, stratify=stratify_temp
        )

        container = DatasetContainer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
            feature_names=X.columns.tolist(),
            target_name=target_column,
            metadata={
                "total_samples": len(df),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "n_features": X.shape[1],
                "file_source": file_path
            }
        )
        
        self.logger.info(f"Successfully loaded and split dataset. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return container
