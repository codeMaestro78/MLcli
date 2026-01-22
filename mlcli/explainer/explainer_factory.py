"""
Explainer Factory

Factory for creating model explainers.
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExplainerFactory:
    """
    Factory for creating model explainers.

    Supports SHAP and LIME explainers.
    """

    _methods = {
        "shap": {
            "name": "SHAP",
            "full_name": "SHapley Additive exPlanations",
            "description": "Game theory approach to explain model predictions",
            "best_for": "Tree-based models, global explanations",
        },
        "lime": {
            "name": "LIME",
            "full_name": "Local Interpretable Model-agnostic Explanations",
            "description": "Local surrogate model for instance explanations",
            "best_for": "Any model, local explanations",
        },
    }

    @classmethod
    def create(
        cls,
        method: str,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """
        Create an explainer instance.

        Args:
            method: Explainer method ('shap' or 'lime')
            model: Trained model to explain
            feature_names: Names of features
            class_names: Names of target classes
            **kwargs: Additional arguments for explainer

        Returns:
            Explainer instance
        """
        method = method.lower()

        if method not in cls._methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(cls._methods.keys())}")

        if method == "shap":
            from mlcli.explainer.shap_explainer import SHAPExplainer

            return SHAPExplainer(
                model=model,
                feature_names=feature_names,
                class_names=class_names,
                explainer_type=kwargs.get("explainer_type", "auto"),
            )

        elif method == "lime":
            from mlcli.explainer.lime_explainer import LIMEExplainer

            return LIMEExplainer(
                model=model,
                feature_names=feature_names,
                class_names=class_names,
                mode=kwargs.get("mode", "classification"),
            )

    @classmethod
    def list_methods(cls) -> List[str]:
        """
        List available explainer methods.

        Returns:
            List of method names
        """
        return list(cls._methods.keys())

    @classmethod
    def get_method_info(cls, method: str) -> Dict[str, str]:
        """
        Get information about an explainer method.

        Args:
            method: Method name

        Returns:
            Dictionary with method information
        """
        method = method.lower()
        if method not in cls._methods:
            raise ValueError(f"Unknown method: {method}")
        return cls._methods[method]


def get_explainer(
    method: str,
    model: Any,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    **kwargs,
) -> Any:
    """
    Convenience function to create an explainer.

    Args:
        method: Explainer method ('shap' or 'lime')
        model: Trained model to explain
        feature_names: Names of features
        class_names: Names of target classes
        **kwargs: Additional arguments

    Returns:
        Explainer instance
    """
    return ExplainerFactory.create(
        method=method, model=model, feature_names=feature_names, class_names=class_names, **kwargs
    )
