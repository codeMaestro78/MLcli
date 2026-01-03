"""
AutoML Module

Automated Machine Learning for model selection, hyperparameter tuning,
and preprocessing. Provides a unified interface for automatic model
training and optimization.
"""

from mlcli.automl.search_space import SearchSpaceGenerator
from mlcli.automl.model_selector import ModelSelector
from mlcli.automl.base_automl import BaseAutoML
from mlcli.automl.automl_classifier import AutoMLClassifier
from mlcli.automl.data_analyzer import DataAnalyzer
from mlcli.automl.preprocessing_selector import PreprocessingSelector
from mlcli.automl.reporter import AutoMLReporter

__all__=[
    "BaseAutoML",
    "AutoMLClassifier",
    "ModelSelector",
    "SearchSpaceGenerator",
    "DataAnalyzer",
    "PreProcessingSelector",
    "AutoMLReporter"
]
