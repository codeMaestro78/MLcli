"""
Encoder Preprocessors

Label, OneHot, and Ordinal encoding for categorical features.
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import logging

from mlcli.preprocessor.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class LabelEncoderProcessor(BasePreprocessor):
    """
    LabelEncoder - Encode target labels with values between 0 and n_classes-1.

    Useful for encoding categorical target variables.
    """

    def __init__(self, **kwargs):
        """
        Initialize LabelEncoder.

        Args:
            **kwargs: Additional configuration
        """
        super().__init__(name="label_encoder", **kwargs)
        self._preprocessor = LabelEncoder()
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LabelEncoderProcessor":
        """
        Fit the encoder to labels.

        Args:
            X: Labels to encode (1D array)
            y: Ignored

        Returns:
            Self
        """
        # LabelEncoder works on 1D arrays
        if X.ndim > 1:
            X = X.ravel()

        logger.info(f"Fitting LabelEncoder on {len(X)} labels")
        self._preprocessor.fit(X)
        self._classes = self._preprocessor.classes_
        self._fitted = True

        logger.info(f"LabelEncoder fitted. Classes: {self._classes}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform labels to encoded values.

        Args:
            X: Labels to encode (1D array)

        Returns:
            Encoded labels
        """
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before transform")

        if X.ndim > 1:
            X = X.ravel()

        return self._preprocessor.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded labels.

        Args:
            X: Encoded labels

        Returns:
            Original labels
        """
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before inverse_transform")

        if X.ndim > 1:
            X = X.ravel()

        return self._preprocessor.inverse_transform(X)

    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        params = super().get_params()
        if self._fitted:
            params["classes"] = self._classes.tolist()
            params["n_classes"] = len(self._classes)
        return params


class OneHotEncoderProcessor(BasePreprocessor):
    """
    OneHotEncoder - Encode categorical features as one-hot numeric arrays.

    Creates binary columns for each category.
    """

    def __init__(
        self,
        categories: str = "auto",
        drop: Optional[str] = None,
        sparse_output: bool = False,
        handle_unknown: str = "error",
        **kwargs,
    ):
        """
        Initialize OneHotEncoder.

        Args:
            categories: 'auto' or list of arrays with categories
            drop: None, 'first', or 'if_binary' to drop one category
            sparse_output: If True, return sparse matrix
            handle_unknown: 'error' or 'ignore' for unknown categories
            **kwargs: Additional configuration
        """
        super().__init__(name="onehot_encoder", **kwargs)
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self._preprocessor = OneHotEncoder(
            categories=categories,
            drop=drop,
            sparse_output=sparse_output,
            handle_unknown=handle_unknown,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OneHotEncoderProcessor":
        """
        Fit the encoder to data.

        Args:
            X: Feature data (2D array)
            y: Ignored

        Returns:
            Self
        """
        logger.info(f"Fitting OneHotEncoder on data with shape: {X.shape}")
        self._preprocessor.fit(X)
        self._fitted = True

        # Set feature names
        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        # Generate output feature names
        self._feature_names_out = []
        for i, cats in enumerate(self._preprocessor.categories_):
            feature_name = (
                self._feature_names_in[i] if i < len(self._feature_names_in) else f"feature_{i}"
            )
            for cat in cats:
                self._feature_names_out.append(f"{feature_name}_{cat}")

        logger.info(f"OneHotEncoder fitted. Output features: {len(self._feature_names_out)}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to one-hot encoded format.

        Args:
            X: Feature data to encode

        Returns:
            One-hot encoded data
        """
        if not self._fitted:
            raise ValueError("OneHotEncoder must be fitted before transform")

        result = self._preprocessor.transform(X)

        # Convert to dense array if sparse
        if hasattr(result, "toarray"):
            result = result.toarray()

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform one-hot encoded data."""
        if not self._fitted:
            raise ValueError("OneHotEncoder must be fitted before inverse_transform")

        return self._preprocessor.inverse_transform(X)

    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        params = super().get_params()
        if self._fitted:
            params["categories"] = [cats.tolist() for cats in self._preprocessor.categories_]
            params["n_features_out"] = len(self._feature_names_out)
        return params


class OrdinalEncoderProcessor(BasePreprocessor):
    """
    OrdinalEncoder - Encode categorical features as ordinal integers.

    Each category is mapped to an integer.
    """

    def __init__(
        self,
        categories: str = "auto",
        handle_unknown: str = "error",
        unknown_value: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize OrdinalEncoder.

        Args:
            categories: 'auto' or list of arrays with categories
            handle_unknown: 'error' or 'use_encoded_value'
            unknown_value: Value to use for unknown categories
            **kwargs: Additional configuration
        """
        super().__init__(name="ordinal_encoder", **kwargs)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self._preprocessor = OrdinalEncoder(
            categories=categories, handle_unknown=handle_unknown, unknown_value=unknown_value
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OrdinalEncoderProcessor":
        """
        Fit the encoder to data.

        Args:
            X: Feature data (2D array)
            y: Ignored

        Returns:
            Self
        """
        logger.info(f"Fitting OrdinalEncoder on data with shape: {X.shape}")
        self._preprocessor.fit(X)
        self._fitted = True

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]
        self._feature_names_out = self._feature_names_in.copy()

        logger.info(f"OrdinalEncoder fitted. Features: {X.shape[1]}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to ordinal encoded format."""
        if not self._fitted:
            raise ValueError("OrdinalEncoder must be fitted before transform")

        return self._preprocessor.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform ordinal encoded data."""
        if not self._fitted:
            raise ValueError("OrdinalEncoder must be fitted before inverse_transform")

        return self._preprocessor.inverse_transform(X)

    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        params = super().get_params()
        if self._fitted:
            params["categories"] = [cats.tolist() for cats in self._preprocessor.categories_]
        return params
