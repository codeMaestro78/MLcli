"""
Normalizer Preprocessor

Normalize samples to unit norm (L1, L2, Max).
"""

from typing import Optional
import numpy as np
from sklearn.preprocessing import Normalizer
import logging

from mlcli.preprocessor.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class NormalizerProcessor(BasePreprocessor):
    """
    Normalizer - Normalize samples individually to unit norm.

    Each sample (row) is scaled to have unit norm.
    Supported norms: l1, l2, max
    """

    def __init__(self, norm: str = "l2", **kwargs):
        """
        Initialize Normalizer.

        Args:
            norm: Type of norm to use ('l1', 'l2', 'max')
            **kwargs: Additional configuration
        """
        if norm not in ["l1", "l2", "max"]:
            raise ValueError(f"norm must be 'l1', 'l2', or 'max', got: {norm}")

        super().__init__(name="normalizer", **kwargs)
        self.norm = norm
        self._preprocessor = Normalizer(norm=norm)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NormalizerProcessor":
        """
        Fit the normalizer (stateless, but kept for API consistency).

        Args:
            X: Feature data
            y: Ignored

        Returns:
            Self
        """
        logger.info(f"Fitting Normalizer (norm={self.norm}) on data with shape: {X.shape}")
        self._preprocessor.fit(X)
        self._fitted = True

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]
        self._feature_names_out = self._feature_names_in.copy()

        logger.info(f"Normalizer fitted with {self.norm} norm")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using normalizer.

        Args:
            X: Feature data to transform

        Returns:
            Normalized data
        """
        if not self._fitted:
            raise ValueError("Normalizer must be fitted before transform")

        logger.debug(f"Normalizing data with shape: {X.shape}")
        return self._preprocessor.transform(X)

    def get_params(self):
        """Get normalizer parameters."""
        params = super().get_params()
        params["norm"] = self.norm
        return params

    def get_info(self):
        """Get normalizer information."""
        info = super().get_info()
        info["norm_type"] = self.norm
        return info
