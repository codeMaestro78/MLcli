"""
Base Preprocessor Class

Abstract base class for all preprocessors.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import logging

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    def __init__(self, name: str, **kwargs):
        """
        Initialize preprocessor.

        Args:
            name: Name of the preprocessor
            **kwargs: Additional configuration options
        """
        self.name = name
        self.config = kwargs
        self._fitted = False
        self._preprocessor = None
        self._feature_names_in: Optional[List[str]] = None
        self._feature_names_out: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._fitted

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BasePreprocessor":
        """
        Fit the preprocessor to data.

        Args:
            X: Feature data
            y: Target data (optional, used for supervised methods)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Feature data to transform

        Returns:
            Transformed data
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform data.

        Args:
            X: Feature data
            y: Target data (optional)

        Returns:
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data (if supported).

        Args:
            X: Transformed data

        Returns:
            Original data

        Raises:
            NotImplementedError: If inverse transform is not supported
        """
        raise NotImplementedError(f"Inverse transform not supported for {self.name}")

    def get_params(self) -> Dict[str, Any]:
        """Get preprocessor parameters."""
        return {"name": self.name, "config": self.config, "is_fitted": self._fitted}

    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set input feature names.

        Args:
            feature_names: List of feature names
        """
        self._feature_names_in = feature_names

    def get_feature_names_out(self) -> Optional[List[str]]:
        """
        Get output feature names.

        Returns:
            List of output feature names or None
        """
        return self._feature_names_out

    def save(self, save_path: Union[str, Path]) -> Path:
        """
        Save preprocessor to disk.

        Args:
            save_path: Path to save preprocessor

        Returns:
            Path where preprocessor was saved
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        preprocessor_data = {
            "name": self.name,
            "config": self.config,
            "fitted": self._fitted,
            "preprocessor": self._preprocessor,
            "feature_names_in": self._feature_names_in,
            "feature_names_out": self._feature_names_out,
        }

        with open(save_path, "wb") as f:
            pickle.dump(preprocessor_data, f)

        logger.info(f"Saved preprocessor to: {save_path}")
        return save_path

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> "BasePreprocessor":
        """
        Load preprocessor from disk.

        Args:
            load_path: Path to saved preprocessor

        Returns:
            Loaded preprocessor instance
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        # Create instance
        instance = cls.__new__(cls)
        instance.name = data["name"]
        instance.config = data["config"]
        instance._fitted = data["fitted"]
        instance._preprocessor = data["preprocessor"]
        instance._feature_names_in = data["feature_names_in"]
        instance._feature_names_out = data["feature_names_out"]

        logger.info(f"Loaded preprocessor from: {load_path}")
        return instance

    def get_info(self) -> Dict[str, Any]:
        """Get preprocessor information."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "is_fitted": self._fitted,
            "config": self.config,
            "n_features_in": len(self._feature_names_in) if self._feature_names_in else None,
            "n_features_out": len(self._feature_names_out) if self._feature_names_out else None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self._fitted})"
