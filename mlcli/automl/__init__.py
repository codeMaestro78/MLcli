"""
AutoML Module

Automated Machine Learning for model selection, hyperparameter tuning,
and preprocessing. Provides a unified interface for automatic model
training and optimization.
"""

from mlcli.automl.search_space import SearchSpaceGenerator
from mlcli.automl.model_selector import ModelSelector, ModelCandidate
from mlcli.automl.base_automl import BaseAutoML, LeaderboardEntry
from mlcli.automl.data_analyzer import DataAnalyzer, DataAnalysisReport, ColumnInfo
from mlcli.automl.preprocessing_selector import (
    PreprocessingSelector,
    PreprocessingPlan,
    PreprocessingRecommendation,
)

#  Main AutoML classifier implementation
from mlcli.automl.automl_classifier import AutoMLClassifier

# Step 8: AutoML reporting
from mlcli.automl.reporter import AutoMLReporter

__all__ = [
    # Main AutoML class
    "AutoMLClassifier",
    # Reporting (Step 8)
    "AutoMLReporter",
    # Core classes
    "BaseAutoML",
    "LeaderboardEntry",
    # Model selection
    "ModelSelector",
    "ModelCandidate",
    # Search space
    "SearchSpaceGenerator",
    # Data analysis
    "DataAnalyzer",
    "DataAnalysisReport",
    "ColumnInfo",
    # Preprocessing
    "PreprocessingSelector",
    "PreprocessingPlan",
    "PreprocessingRecommendation",
]
