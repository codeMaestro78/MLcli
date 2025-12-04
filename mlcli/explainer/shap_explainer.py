"""
SHAP Explainer

SHAP (SHapley Additive exPlanations) implementation for model interpretability.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from mlcli.explainer.base_explainer import BaseExplainer

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based model explainer.

    Uses SHAP values to explain feature contributions to predictions.
    Supports Tree, Kernel, and Linear explainers based on model type.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        explainer_type: str = "auto",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model
            feature_names: Names of features
            class_names: Names of target classes
            explainer_type: Type of SHAP explainer
                           ('auto', 'tree', 'kernel', 'linear')
        """
        super().__init__(model, feature_names, class_names)
        self.explainer_type = explainer_type
        self.shap_explainer = None
        self.shap_values = None
        self.expected_value = None
        self.background_data = None

    def _create_explainer(self, X_background: Optional[np.ndarray] = None) -> None:
        """
        Create appropriate SHAP explainer based on model type.

        Args:
            X_background: Background data for Kernel explainer
        """
        import shap

        model_type = type(self.model).__name__.lower()

        if self.explainer_type == "auto":
            # Auto-detect appropriate explainer
            if any(
                tree_type in model_type
                for tree_type in ["randomforest", "xgb", "lgbm", "catboost", "gradient", "tree"]
            ):
                self.explainer_type = "tree"
            elif "linear" in model_type or "logistic" in model_type:
                self.explainer_type = "linear"
            else:
                self.explainer_type = "kernel"

        logger.info(f"Using SHAP {self.explainer_type} explainer for {model_type}")

        if self.explainer_type == "tree":
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"TreeExplainer failed: {e}. Falling back to Kernel.")
                self.explainer_type = "kernel"

        if self.explainer_type == "linear":
            try:
                if X_background is not None:
                    self.shap_explainer = shap.LinearExplainer(self.model, X_background)
                else:
                    raise ValueError("Background data required for LinearExplainer")
            except Exception as e:
                logger.warning(f"LinearExplainer failed: {e}. Falling back to Kernel.")
                self.explainer_type = "kernel"

        if self.explainer_type == "kernel":
            if X_background is None:
                raise ValueError("Background data required for KernelExplainer")

            # Use predict_proba if available for classification
            predict_fn = self._get_predict_function()

            # Sample background data if too large
            if len(X_background) > 100:
                indices = np.random.choice(len(X_background), 100, replace=False)
                X_background = X_background[indices]

            self.background_data = X_background
            self.shap_explainer = shap.KernelExplainer(predict_fn, X_background)

    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        X_background: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        max_samples: int = 500,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.

        Args:
            X: Input data to explain
            X_background: Background data for explainer
            max_samples: Maximum samples to explain
            **kwargs: Additional arguments

        Returns:
            Dictionary containing SHAP values and explanations
        """
        import shap

        X = self._validate_input(X)

        if X_background is not None:
            X_background = self._validate_input(X_background)
        else:
            X_background = X

        # Limit samples
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]

        # Create explainer if not exists
        if self.shap_explainer is None:
            self._create_explainer(X_background)

        logger.info(f"Computing SHAP values for {len(X)} samples...")

        # Compute SHAP values
        self.shap_values = self.shap_explainer.shap_values(X)
        self.expected_value = self.shap_explainer.expected_value

        n_features = X.shape[1]

        # Handle different SHAP output formats
        if isinstance(self.shap_values, list):
            # Multi-class: list of arrays, one per class
            # Each element has shape (n_samples, n_features)
            # For binary classification, use class 1 (positive class)
            if len(self.shap_values) == 2:
                # Binary classification - use positive class
                shap_values_for_importance = np.abs(self.shap_values[1])
            else:
                # Multi-class - average across all classes
                stacked = np.array(self.shap_values)  # (n_classes, n_samples, n_features)
                shap_values_for_importance = np.abs(stacked).mean(axis=0)  # (n_samples, n_features)

            # Mean across samples to get feature importance
            shap_values_mean = shap_values_for_importance.mean(axis=0)  # (n_features,)
        else:
            # Single output, regression, or newer SHAP format
            shap_array = np.asarray(self.shap_values)

            # Check if shape matches expected
            if shap_array.ndim == 2:
                if shap_array.shape[1] == n_features:
                    # Shape is (n_samples, n_features) - correct
                    shap_values_mean = np.abs(shap_array).mean(axis=0)
                elif shap_array.shape[1] == n_features * 2:
                    # Shape is (n_samples, n_features*2) - binary class stacked
                    # Take the second half (positive class)
                    shap_values_mean = np.abs(shap_array[:, n_features:]).mean(axis=0)
                else:
                    # Unknown format - just average
                    shap_values_mean = np.abs(shap_array).mean(axis=0)
            elif shap_array.ndim == 3:
                # Shape is (n_samples, n_features, n_classes)
                # Use positive class for binary or average for multi-class
                if shap_array.shape[2] == 2:
                    shap_values_mean = np.abs(shap_array[:, :, 1]).mean(axis=0)
                else:
                    shap_values_mean = np.abs(shap_array).mean(axis=(0, 2))
            else:
                shap_values_mean = np.abs(shap_array).mean(axis=0)

        # Ensure shap_values_mean is 1D with correct length
        shap_values_mean = np.asarray(shap_values_mean).flatten()

        # If still wrong length, truncate or pad
        if len(shap_values_mean) != n_features:
            logger.warning(
                f"SHAP values length ({len(shap_values_mean)}) doesn't match "
                f"features ({n_features}). Taking first {n_features} values."
            )
            shap_values_mean = shap_values_mean[:n_features]

        # Create feature importance ranking
        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure feature_names matches the number of features
        if len(feature_names) != len(shap_values_mean):
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match "
                f"SHAP values length ({len(shap_values_mean)}). Using indices."
            )
            feature_names = [f"feature_{i}" for i in range(len(shap_values_mean))]

        # Convert to float for JSON serialization
        importance_dict = {name: float(val) for name, val in zip(feature_names, shap_values_mean)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        self.explanations = {
            "method": "shap",
            "explainer_type": self.explainer_type,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_importance": sorted_importance,
            "expected_value": (
                self.expected_value.tolist()
                if isinstance(self.expected_value, np.ndarray)
                else self.expected_value
            ),
            "top_features": list(sorted_importance.keys())[:10],
        }

        logger.info("SHAP explanation complete")

        return self.explanations

    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        X_background: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance.

        Args:
            instance: Single data point to explain
            X_background: Background data for explainer
            **kwargs: Additional arguments

        Returns:
            Dictionary containing instance explanation
        """
        if isinstance(instance, pd.Series):
            instance = instance.values

        instance = np.asarray(instance).reshape(1, -1)
        n_features = instance.shape[1]

        if self.shap_explainer is None:
            if X_background is None:
                raise ValueError("Background data required for first explanation")
            self._create_explainer(X_background)

        # Get SHAP values for instance
        instance_shap = self.shap_explainer.shap_values(instance)

        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        # Handle different output formats
        if isinstance(instance_shap, list):
            # Multi-class list format
            if len(instance_shap) == 2:
                # Binary classification - use positive class
                shap_vals = np.asarray(instance_shap[1]).flatten()
            else:
                # Multi-class - use the class with highest values
                stacked = np.array([np.asarray(sv).flatten() for sv in instance_shap])
                pred_class = np.argmax(np.abs(stacked).sum(axis=1))
                shap_vals = stacked[pred_class]
        else:
            # Single array format
            shap_array = np.asarray(instance_shap).flatten()

            # If twice the features, take second half (positive class)
            if len(shap_array) == n_features * 2:
                shap_vals = shap_array[n_features:]
            elif len(shap_array) >= n_features:
                shap_vals = shap_array[:n_features]
            else:
                shap_vals = shap_array

        # Ensure correct length
        if len(shap_vals) != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

        # Convert to float values
        contributions = {name: float(val) for name, val in zip(feature_names, shap_vals)}
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "method": "shap",
            "instance_values": instance[0].tolist(),
            "feature_contributions": sorted_contributions,
            "expected_value": (
                self.expected_value.tolist()
                if isinstance(self.expected_value, np.ndarray)
                else self.expected_value
            ),
            "top_positive_features": [k for k, v in sorted_contributions.items() if v > 0][:5],
            "top_negative_features": [k for k, v in sorted_contributions.items() if v < 0][:5],
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from SHAP values.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.explanations is None:
            raise RuntimeError("Run explain() first to compute feature importance")

        return self.explanations.get("feature_importance", {})

    def plot(
        self,
        plot_type: str = "summary",
        output_path: Optional[Path] = None,
        max_display: int = 20,
        **kwargs,
    ) -> Optional[str]:
        """
        Generate SHAP plots.

        Args:
            plot_type: Type of plot ('summary', 'bar', 'beeswarm', 'waterfall')
            output_path: Path to save the plot
            max_display: Maximum features to display
            **kwargs: Additional plot arguments

        Returns:
            Path to saved plot or None
        """
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.shap_values is None:
            raise RuntimeError("Run explain() first to generate plots")

        plt.figure(figsize=(12, 8))

        # Get data for plotting
        X_plot = kwargs.get("X")

        shap_values_plot = self.shap_values
        if isinstance(shap_values_plot, list):
            # For multi-class, use first class or average
            if len(shap_values_plot) == 2:
                shap_values_plot = shap_values_plot[1]
            else:
                shap_values_plot = np.mean(np.abs(shap_values_plot), axis=0)

        if plot_type == "summary":
            if X_plot is not None:
                shap.summary_plot(
                    shap_values_plot,
                    X_plot,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False,
                )
            else:
                # Bar plot if no X data
                shap.summary_plot(
                    shap_values_plot,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    plot_type="bar",
                    show=False,
                )

        elif plot_type == "bar":
            importance = self.get_feature_importance()
            features = list(importance.keys())[:max_display]
            values = [importance[f] for f in features]

            plt.barh(features[::-1], values[::-1], color="steelblue")
            plt.xlabel("Mean |SHAP Value|")
            plt.title("Feature Importance (SHAP)")
            plt.tight_layout()

        elif plot_type == "beeswarm":
            if X_plot is not None:
                shap.plots.beeswarm(
                    shap.Explanation(
                        values=shap_values_plot, data=X_plot, feature_names=self.feature_names
                    ),
                    max_display=max_display,
                    show=False,
                )

        elif plot_type == "waterfall":
            instance_idx = kwargs.get("instance_idx", 0)
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_plot[instance_idx],
                    base_values=(
                        self.expected_value
                        if not isinstance(self.expected_value, list)
                        else self.expected_value[0]
                    ),
                    feature_names=self.feature_names,
                ),
                max_display=max_display,
                show=False,
            )

        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Plot saved to {output_path}")
            return str(output_path)

        plt.close()
        return None

    def get_summary_text(self) -> str:
        """
        Get text summary of SHAP explanation.

        Returns:
            Human-readable summary
        """
        if self.explanations is None:
            return "No explanation computed yet. Run explain() first."

        lines = [
            "=" * 60,
            "SHAP Explanation Summary",
            "=" * 60,
            f"Explainer Type: {self.explanations['explainer_type']}",
            f"Samples Analyzed: {self.explanations['n_samples']}",
            f"Features: {self.explanations['n_features']}",
            "",
            "Top 10 Most Important Features:",
            "-" * 40,
        ]

        importance = self.explanations["feature_importance"]
        for i, (feature, value) in enumerate(list(importance.items())[:10], 1):
            lines.append(f"  {i}. {feature}: {value:.4f}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
