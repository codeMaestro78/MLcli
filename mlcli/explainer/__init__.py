"""
MLCLI Model Explainability Module

Provides SHAP and LIME explanations for model predictions.
"""

from mlcli.explainer.base_explainer import BaseExplainer
from mlcli.explainer.shap_explainer import SHAPExplainer
from mlcli.explainer.lime_explainer import LIMEExplainer
from mlcli.explainer.explainer_factory import ExplainerFactory, get_explainer

__all__ = [
    "BaseExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "ExplainerFactory",
    "get_explainer"
]
