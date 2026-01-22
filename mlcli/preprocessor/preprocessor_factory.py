"""
Preprocessor Factory

Factory for creating preprocessors by name.
"""

from typing import Dict, Any, List, Type
import logging

from mlcli.preprocessor.base_preprocessor import BasePreprocessor
from mlcli.preprocessor.scalers import (
    StandardScalerProcessor,
    MinMaxScalerProcessor,
    RobustScalerProcessor,
)
from mlcli.preprocessor.normalizers import NormalizerProcessor
from mlcli.preprocessor.encoders import (
    LabelEncoderProcessor,
    OneHotEncoderProcessor,
    OrdinalEncoderProcessor,
)
from mlcli.preprocessor.feature_selectors import (
    SelectKBestProcessor,
    RFEProcessor,
    VarianceThresholdProcessor,
)

logger = logging.getLogger(__name__)

# Registry of preprocessors
PREPROCESSOR_REGISTRY: Dict[str, Type[BasePreprocessor]] = {
    # Scalers
    "standard_scaler": StandardScalerProcessor,
    "minmax_scaler": MinMaxScalerProcessor,
    "robust_scaler": RobustScalerProcessor,
    # Normalizers
    "normalizer": NormalizerProcessor,
    "l1_normalizer": NormalizerProcessor,
    "l2_normalizer": NormalizerProcessor,
    "max_normalizer": NormalizerProcessor,
    # Encoders
    "label_encoder": LabelEncoderProcessor,
    "onehot_encoder": OneHotEncoderProcessor,
    "ordinal_encoder": OrdinalEncoderProcessor,
    # Feature Selectors
    "select_k_best": SelectKBestProcessor,
    "rfe": RFEProcessor,
    "variance_threshold": VarianceThresholdProcessor,
}

# Preprocessor info for display
PREPROCESSOR_INFO: Dict[str, Dict[str, Any]] = {
    "standard_scaler": {
        "name": "StandardScaler",
        "category": "Scaling",
        "description": "Standardize features by removing mean and scaling to unit variance",
        "params": {"with_mean": True, "with_std": True},
    },
    "minmax_scaler": {
        "name": "MinMaxScaler",
        "category": "Scaling",
        "description": "Scale features to a given range (default 0-1)",
        "params": {"feature_range": (0, 1)},
    },
    "robust_scaler": {
        "name": "RobustScaler",
        "category": "Scaling",
        "description": "Scale features using statistics robust to outliers (median/IQR)",
        "params": {"with_centering": True, "with_scaling": True},
    },
    "normalizer": {
        "name": "Normalizer",
        "category": "Normalization",
        "description": "Normalize samples individually to unit norm",
        "params": {"norm": "l2"},
    },
    "l1_normalizer": {
        "name": "L1 Normalizer",
        "category": "Normalization",
        "description": "Normalize samples to L1 norm (sum of absolute values = 1)",
        "params": {"norm": "l1"},
    },
    "l2_normalizer": {
        "name": "L2 Normalizer",
        "category": "Normalization",
        "description": "Normalize samples to L2 norm (Euclidean norm = 1)",
        "params": {"norm": "l2"},
    },
    "max_normalizer": {
        "name": "Max Normalizer",
        "category": "Normalization",
        "description": "Normalize samples by maximum absolute value",
        "params": {"norm": "max"},
    },
    "label_encoder": {
        "name": "LabelEncoder",
        "category": "Encoding",
        "description": "Encode target labels with values between 0 and n_classes-1",
        "params": {},
    },
    "onehot_encoder": {
        "name": "OneHotEncoder",
        "category": "Encoding",
        "description": "Encode categorical features as one-hot numeric arrays",
        "params": {"sparse_output": False, "handle_unknown": "error"},
    },
    "ordinal_encoder": {
        "name": "OrdinalEncoder",
        "category": "Encoding",
        "description": "Encode categorical features as ordinal integers",
        "params": {"handle_unknown": "error"},
    },
    "select_k_best": {
        "name": "SelectKBest",
        "category": "Feature Selection",
        "description": "Select features according to the k highest scores",
        "params": {"k": 10, "score_func": "f_classif"},
    },
    "rfe": {
        "name": "RFE",
        "category": "Feature Selection",
        "description": "Recursive Feature Elimination based on model importance",
        "params": {"n_features_to_select": 10, "step": 1},
    },
    "variance_threshold": {
        "name": "VarianceThreshold",
        "category": "Feature Selection",
        "description": "Remove features with variance below threshold",
        "params": {"threshold": 0.0},
    },
}


class PreprocessorFactory:
    """Factory for creating preprocessors."""

    @staticmethod
    def create(method: str, **kwargs) -> BasePreprocessor:
        """
        Create a preprocessor by name.

        Args:
            method: Name of preprocessor
            **kwargs: Configuration options

        Returns:
            Preprocessor instance

        Raises:
            ValueError: If method is unknown
        """
        method = method.lower().replace("-", "_")

        if method not in PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Unknown preprocessor: {method}. "
                f"Available: {list(PREPROCESSOR_REGISTRY.keys())}"
            )

        preprocessor_class = PREPROCESSOR_REGISTRY[method]

        # Handle normalizer variants
        if method in ["l1_normalizer", "l2_normalizer", "max_normalizer"]:
            norm = method.replace("_normalizer", "")
            kwargs["norm"] = norm

        logger.info(f"Creating preprocessor: {method}")
        return preprocessor_class(**kwargs)

    @staticmethod
    def list_methods() -> List[str]:
        """List available preprocessor methods."""
        return list(PREPROCESSOR_REGISTRY.keys())

    @staticmethod
    def list_by_category() -> Dict[str, List[str]]:
        """List preprocessors grouped by category."""
        categories: Dict[str, List[str]] = {}
        for method, info in PREPROCESSOR_INFO.items():
            category = info.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(method)
        return categories

    @staticmethod
    def get_method_info(method: str) -> Dict[str, Any]:
        """
        Get information about a preprocessor method.

        Args:
            method: Name of preprocessor

        Returns:
            Dictionary with method information
        """
        method = method.lower().replace("-", "_")
        return PREPROCESSOR_INFO.get(
            method,
            {
                "name": method,
                "category": "Unknown",
                "description": "No description available",
                "params": {},
            },
        )

    @staticmethod
    def get_default_params(method: str) -> Dict[str, Any]:
        """
        Get default parameters for a preprocessor.

        Args:
            method: Name of preprocessor

        Returns:
            Dictionary of default parameters
        """
        info = PreprocessorFactory.get_method_info(method)
        return info.get("params", {})


def get_preprocessor(method: str, **kwargs) -> BasePreprocessor:
    """
    Convenience function to create a preprocessor.

    Args:
        method: Name of preprocessor
        **kwargs: Configuration options

    Returns:
        Preprocessor instance
    """
    return PreprocessorFactory.create(method, **kwargs)
