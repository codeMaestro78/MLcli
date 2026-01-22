"""
Search Space Generator

Provides default hyperparameter search spaces for all supported models.
Spaces are Optuna-compatible and can be adjusted based on data characteristics.
"""

from typing import Dict, Any, Optional, List
import logging
import copy

logger = logging.getLogger(__name__)


DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    # Logistic Regression
    "logistic_regression": {
        "C": {"type": "loguniform", "low": 1e-4, "high": 100.0},
        "penalty": {"type": "categorical", "choices": ["l2"]},
        "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
        "max_iter": {"type": "int", "low": 100, "high": 1000, "step": 100},
    },
    # Support Vector Machine
    "svm": {
        "C": {"type": "loguniform", "low": 1e-3, "high": 100.0},
        "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        "degree": {"type": "int", "low": 2, "high": 5},
    },
    # Random Forest
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
    },
    # XGBoost
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "gamma": {"type": "float", "low": 0.0, "high": 0.5},
    },
    # LightGBM
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
        "num_leaves": {"type": "int", "low": 20, "high": 150},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
    },
    # CatBoost
    "catboost": {
        "iterations": {"type": "int", "low": 100, "high": 500, "step": 50},
        "depth": {"type": "int", "low": 4, "high": 10},
        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
        "l2_leaf_reg": {"type": "loguniform", "low": 1.0, "high": 10.0},
        "border_count": {"type": "int", "low": 32, "high": 255},
    },
}


MODEL_COMPLEXITY: Dict[str, int] = {
    "logistic_regression": 1,
    "svm": 5,
    "random_forest": 4,
    "xgboost": 6,
    "lightgbm": 5,
    "catboost": 7,
}

EARLY_STOPPING_MODELS: List[str] = [
    "xgboost",
    "lightgbm",
    "catboost",
]


class SearchSpaceGenerator:
    """
    Generates hyperparameter search spaces for AutoML.

    Supports:
    - Default Optuna-compatible spaces
    - Partial user overrides
    - Data-size–aware adjustments
    """

    def __init__(
        self,
        custom_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        data_size_adjustment: bool = True,
    ):
        """
        Initialize search space generator.

        Args:
            custom_spaces:
                Optional overrides for specific models.
            data_size_adjustment:
                Whether to adjust spaces based on dataset size.
        """

        # Deep copy prevents shared mutable state bugs
        self.spaces: Dict[str, Dict[str, Any]] = copy.deepcopy(DEFAULT_SEARCH_SPACES)

        if custom_spaces is not None:
            for model, space in custom_spaces.items():
                if model in self.spaces:
                    self.spaces[model].update(space)
                else:
                    self.spaces[model] = copy.deepcopy(space)

        self.data_size_adjustment = data_size_adjustment

        logger.debug(
            "SearchSpaceGenerator initialized | models=%d | data_size_adjustment=%s",
            len(self.spaces),
            self.data_size_adjustment,
        )

    def get_space(
        self, model_name: str, n_samples: Optional[int] = None, n_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get search space for a specific model.

        Args:
            model_name: Name of the model
            n_samples: Number of samples in dataset (for adjustment)
            n_features: Number of features (for adjustment)

        Returns:
            Hyperparameter search space dictionary
        """

        if model_name not in self.spaces:
            logger.warning(f"No deafult space for '{model_name}'. Returning empty space.")
            return {}

        space = copy.deepcopy(self.spaces[model_name])

        # Adjust for data size
        if self.data_size_adjustment and n_samples is not None:
            space = self._adjust_for_data_size(space, model_name, n_samples, n_features)

        logger.debug(f"Generated search space for '{model_name}' with {len(space)} params")
        return space

    def _adjust_for_data_size(
        self,
        space: Dict[str, Any],
        model_name: str,
        n_samples: int,
        n_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Adjust search space based on dataset size.

        Reduces search space for large datasets to save time.
        """

        adjusted = {}

        for param, config in space.items():
            adj_config = copy.deepcopy(config)

            # Large dataset (>100K samples)
            if n_samples > 100000:
                if param == "n_estimators" and config.get("type") == "int":
                    adj_config["high"] = min(config.get("high", 500), 300)
                if param == "max_depth" and config.get("type") == "int":
                    adj_config["high"] = min(config.get("high", 30), 15)

            # Small dataset (<1K samples)
            elif n_samples < 1000:
                if param == "n_estimators" and config.get("type") == "int":
                    adj_config["high"] = min(config.get("high", 500), 200)

            adjusted[param] = adj_config

        return adjusted

    def get_all_spaces(self) -> Dict[str, Dict[str, Any]]:
        # Get all available search spaces.
        return copy.deepcopy(self.spaces)

    def list_models(self) -> List[str]:
        # List all models with defined search spaces
        return list(self.spaces.keys())

    def get_complexity(self, model_name: str) -> int:
        """
        Get relative training complexity for a model.

        Returns:
            Complexity score 1-10 (higher = slower)
        """
        return MODEL_COMPLEXITY.get(model_name, 5)

    def supports_early_stopping(self, model_name: str) -> bool:
        # check if model supports early stopping.
        return model_name in EARLY_STOPPING_MODELS

    @staticmethod
    def merge_spaces(base_space: Dict[str, Any], override_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two search spaces, with override taking precedence.

        Args:
            base_space: Base search space
            override_space: Override values

        Returns:
            Merged search space
        """
        merged = copy.deepcopy(base_space)
        merged.update(override_space)
        return merged

    def __repr__(self) -> str:
        return f"SearchSpaceGenerator(models={list(self.spaces.keys())})"
