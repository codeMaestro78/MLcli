"""
Metrics Computation

Provides unified metrics calculation for both ML and DL models,
supporting classification and regression tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    task: str = "classification",
    average: str = "weighted",
    metrics_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for model evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC calculation)
        task: Type of task ('classification' or 'regression')
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')
        metrics_list: Specific metrics to compute (None = all available)

    Returns:
        Dictionary of metric names to values

    Raises:
        ValueError: If task type is unsupported
    """

    logger.debug(f"Computing {task} metrics")

    if task == "classification":
        return _compute_classification_metrics(y_test, y_pred, y_pred_proba, average, metrics_list)
    elif task == "regression":
        return _compute_regression_metrics(y_test, y_pred, metrics_list)
    else:
        raise ValueError(f"Unsupported task: {task}. Use 'classification' or 'regression' ")


def _compute_classification_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
    metrics_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        average: Averaging method for multi-class
        metrics_list: Specific metrics to compute

    Returns:
        Dictionary of metrics
    """

    metrics = {}

    # Default metrics if not specified
    if metrics_list is None:
        metrics_list = ["accuracy", "precision", "recall", "f1"]
        if y_pred_proba is not None:
            metrics_list.append("auc")

    # Determine if binary or multiclass
    n_classes = len(np.unique(y_test))
    is_binary = n_classes = 2

    try:
        # Accuracy
        if "accuracy" in metrics_list:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)

        # Precision
        if "precision" in metrics_list:
            metrics["precision"] = precision_score(y_test, y_pred, average=average, zero_division=0)

        # Recall
        if "recall" in metrics_list:
            metrics["recall"] = recall_score(y_test, y_pred, average=average, zero_division=0)

        # F1 Score
        if "f1" in metrics_list:
            metrics["f1"] = f1_score(y_test, y_pred, average=average, zero_division=0)

        # AUC (only if probabilities provided)
        if "auc" in metrics_list and y_pred_proba is not None:
            try:
                if is_binary:
                    # Binary classification - use probabilities for positive class
                    if y_pred_proba.ndim == 1:
                        proba = y_pred_proba

                    else:
                        proba = y_pred_proba[:, 1]
                    metrics["auc"] = roc_auc_score(y_test, proba)
                else:
                    # Multi-class - use one-vs-rest
                    metrics["auc"] = roc_auc_score(
                        y_test, y_pred_proba, multi_class="ovr", average=average
                    )
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics["auc"] = 0.0

        logger.info(f"Computed metrics: {list(metrics.keys())}")
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise

    return metrics


def _compute_regression_metrics(
    y_test: np.ndarray, y_pred: np.ndarray, metrics_list: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        metrics_list: Specific metrics to compute

    Returns:
        Dictionary of metrics
    """

    metrics = {}

    # Default metrics if not specified

    if metrics_list is None:
        metrics_list = ["mse", "rmse", "mae", "r2"]

    try:
        # Mean sqaured error
        if "mse" in metrics_list:
            metrics["mse"] = mean_squared_error(y_test, y_pred)

        #  Root mean squared error
        if "rmse" in metrics_list:
            metrics["rmse"] = root_mean_squared_error(y_test, y_pred)

        # Mean absolute error
        if "mae" in metrics_list:
            metrics["mae"] = mean_absolute_error(y_test, y_pred)

        #  R^2 Score
        if "r2" in metrics_list:
            metrics["r2"] = r2_score(y_test, y_pred)

        logger.info(f"Computing regression matrix: {list(metrics.keys())}")

    except Exception as e:
        logger.error(f"Error computing regression matrix: {e}")
        raise

    return metrics


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as readable string.

    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places

    Returns:
        Formatted string
    """

    lines = ["Metrics:"]
    lines.append("-" * 40)

    for name, value in metrics.items():
        lines.append(f"  {name:.<20} {value:.{precision}}")

    lines.append("-" * 40)

    return "\n".join(lines)


def get_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, labels: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_test, y_pred, labels=labels)


def get_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False,
) -> Union[str, Dict]:
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        output_dict: Return as dictionary instead of string

    Returns:
        Classification report as string or dictionary
    """
    return classification_report(
        y_test, y_pred, target_names=target_names, output_dict=output_dict, zero_division=0
    )


def compute_per_class_metrics(
    y_test: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class separately.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes

    Returns:
        Dictionary mapping class names to their metrics
    """

    # Get unique classes

    classes = np.unique(y_test)

    if class_names is None:
        class_names = {f"class_{i}" for i in classes}

    per_class_metrics = {}

    for cls, name in zip(classes, class_names):
        #  Create binary labels for this class
        y_test_binary = (y_test == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics = {
            "precision": precision_score(y_test_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_test_binary, y_pred_binary, zero_division=0),
            "f1": f1_score(y_test_binary, y_pred_binary, zero_division=0),
        }
        per_class_metrics[name] = metrics
    return per_class_metrics


def calculate_metric_summary(
    metrics_history: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate summary statistics from multiple metric evaluations.

    Args:
        metrics_history: List of metric dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """

    if not metrics_history:
        return {}

    summary = {}
    metrics_names = metrics_history[0].keys()

    for name in metrics_names:
        values = [m[name] for m in metrics_history if name in m]

        summary[name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }
    return summary
