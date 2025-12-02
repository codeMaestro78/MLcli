"""
Scaler Preprocessors

Standard, MinMax, and Robust scaling for features.
"""

from typing import Optional, List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

from mlcli.preprocessor.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class StandardScalerProcessor(BasePreprocessor):
    """
    StandardScaler - Standardize features by removing mean and scaling to unit variance.

    Z = (X - mean) / std
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, **kwargs):
        """
        Initialize StandardScaler.

        Args:
            with_mean: If True, center data before scaling
            with_std: If True, scale data to unit variance
            **kwargs: Additional configuration
        """
        super().__init__(name="standard_scaler", **kwargs)
        self.with_mean = with_mean
        self.with_std = with_std
        self._preprocessor = StandardScaler(
            with_mean=with_mean,
            with_std=with_std
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "StandardScalerProcessor":
        """
        Fit the scaler to data.

        Args:
            X: Feature data
            y: Ignored (kept for API consistency)

        Returns:
            Self
        """
        logger.info(f"Fitting StandardScaler on data with shape: {X.shape}")
        self._preprocessor.fit(X)
        self._fitted = True

        # Set feature names
        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]
        self._feature_names_out = self._feature_names_in.copy()

        logger.info(f"StandardScaler fitted. Mean: {self._preprocessor.mean_[:3]}...")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Feature data to transform

        Returns:
            Transformed data
        """
        if not self._fitted:
            raise ValueError("StandardScaler must be fitted before transform")

        logger.debug(f"Transforming data with shape: {X.shape}")
        return self._preprocessor.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.

        Args:
            X: Scaled data

        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("StandardScaler must be fitted before inverse_transform")

        return self._preprocessor.inverse_transform(X)

    def get_params(self):
        """Get scaler parameters."""
        params = super().get_params()
        if self._fitted:
            params.update({
                "mean": self._preprocessor.mean_.tolist(),
                "scale": self._preprocessor.scale_.tolist(),
                "var": self._preprocessor.var_.tolist()
            })
        return params


class MinMaxScalerProcessor(BasePreprocessor):
    """
    MinMaxScaler - Scale features to a given range (default 0-1).

    X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    """

    def __init__(self, feature_range: tuple = (0, 1), **kwargs):
        """
        Initialize MinMaxScaler.

        Args:
            feature_range: Desired range of transformed data (min, max)
            **kwargs: Additional configuration
        """
        super().__init__(name="minmax_scaler", **kwargs)
        self.feature_range = feature_range
        self._preprocessor = MinMaxScaler(feature_range=feature_range)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MinMaxScalerProcessor":
        """
        Fit the scaler to data.

        Args:
            X: Feature data
            y: Ignored

        Returns:
            Self
        """
        logger.info(f"Fitting MinMaxScaler on data with shape: {X.shape}")
        self._preprocessor.fit(X)
        self._fitted = True

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]
        self._feature_names_out = self._feature_names_in.copy()

        logger.info(f"MinMaxScaler fitted. Range: {self.feature_range}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self._fitted:
            raise ValueError("MinMaxScaler must be fitted before transform")

        return self._preprocessor.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self._fitted:
            raise ValueError("MinMaxScaler must be fitted before inverse_transform")

        return self._preprocessor.inverse_transform(X)

    def get_params(self):
        """Get scaler parameters."""
        params = super().get_params()
        if self._fitted:
            params.update({
                "data_min": self._preprocessor.data_min_.tolist(),
                "data_max": self._preprocessor.data_max_.tolist(),
                "data_range": self._preprocessor.data_range_.tolist(),
                "scale": self._preprocessor.scale_.tolist()
            })
        return params


class RobustScalerProcessor(BasePreprocessor):
    """
    RobustScaler - Scale features using statistics robust to outliers.

    Uses median and IQR instead of mean and std.
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
        **kwargs
    ):
        """
        Initialize RobustScaler.

        Args:
            with_centering: If True, center data before scaling
            with_scaling: If True, scale data to IQR
            quantile_range: Quantile range for calculating scale
            **kwargs: Additional configuration
        """
        super().__init__(name="robust_scaler", **kwargs)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self._preprocessor = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RobustScalerProcessor":
        """
        Fit the scaler to data.

        Args:
            X: Feature data
            y: Ignored

        Returns:
            Self
        """
        logger.info(f"Fitting RobustScaler on data with shape: {X.shape}")
        self._preprocessor.fit(X)
        self._fitted = True

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]
        self._feature_names_out = self._feature_names_in.copy()

        logger.info(f"RobustScaler fitted. Quantile range: {self.quantile_range}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self._fitted:
            raise ValueError("RobustScaler must be fitted before transform")

        return self._preprocessor.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self._fitted:
            raise ValueError("RobustScaler must be fitted before inverse_transform")

        return self._preprocessor.inverse_transform(X)

    def get_params(self):
        """Get scaler parameters."""
        params = super().get_params()
        if self._fitted:
            params.update({
                "center": self._preprocessor.center_.tolist() if self._preprocessor.center_ is not None else None,
                "scale": self._preprocessor.scale_.tolist() if self._preprocessor.scale_ is not None else None
            })
        return params
