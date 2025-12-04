"""
Base Explainer Class

Abstract base class for all model explainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


class BaseExplainer(ABC):
    """
    Abstract base class for model explainers.

    Provides a common interface for different explanation methods
    like SHAP and LIME.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize explainer.

        Args:
            model: Trained model or predict function
            feature_names: Names of features
            class_names: Names of target classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explanations = None

    @abstractmethod
    def explain(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Generate explanations for predictions.

        Args:
            X: Input data to explain
            **kwargs: Additional arguments

        Returns:
            Dictionary containing explanation results
        """

    @abstractmethod
    def explain_instance(self, instance: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Generate explanation for a single instance.

        Args:
            instance: Single data point to explain
            **kwargs: Additional arguments

        Returns:
            Dictionary containing explanation results
        """

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """

    @abstractmethod
    def plot(
        self, plot_type: str = "summary", output_path: Optional[Path] = None, **kwargs
    ) -> None:
        """
        Generate explanation plots.

        Args:
            plot_type: Type of plot to generate
            output_path: Path to save the plot
            **kwargs: Additional plot arguments
        """

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate and convert input data.

        Args:
            X: Input data

        Returns:
            Numpy array of input data
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            return X.values
        return np.asarray(X)

    def _get_predict_function(self) -> callable:
        """
        Get prediction function from model.

        Returns:
            Callable prediction function
        """
        # Check if model has predict_proba (classification)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        # Check for predict method
        elif hasattr(self.model, "predict"):
            return self.model.predict
        # Model itself might be a function
        elif callable(self.model):
            return self.model
        else:
            raise ValueError("Model must have predict/predict_proba method or be callable")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert explanation results to dictionary.

        Returns:
            Dictionary representation of explanations
        """
        if self.explanations is None:
            return {}
        return self.explanations

    def save_explanation(self, output_path: Path, format: str = "json") -> None:
        """
        Save explanation results to file.

        Args:
            output_path: Path to save results
            format: Output format (json or csv)
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Convert numpy arrays to lists for JSON serialization
            export_data = self._serialize_for_json(self.explanations)
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format == "csv":
            importance = self.get_feature_importance()
            df = pd.DataFrame([importance])
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _serialize_for_json(self, data: Any) -> Any:
        """
        Recursively convert numpy types to JSON-serializable types.

        Args:
            data: Data to serialize

        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        return data
