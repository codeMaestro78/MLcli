"""
Preprocessing Pipeline

Chain multiple preprocessors together.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import logging

from mlcli.preprocessor.base_preprocessor import BasePreprocessor
from mlcli.preprocessor.preprocessor_factory import PreprocessorFactory

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Chain multiple preprocessors into a single pipeline.

    Applies preprocessors in sequence during fit_transform
    and transform operations.
    """

    def __init__(self, steps: Optional[List[Tuple[str, BasePreprocessor]]] = None):
        """
        Initialize pipeline.

        Args:
            steps: List of (name, preprocessor) tuples
        """
        self.steps: List[Tuple[str, BasePreprocessor]] = steps or []
        self._fitted = False
        self._feature_names_in: Optional[List[str]] = None
        self._feature_names_out: Optional[List[str]] = None

    def add_step(self, name: str, preprocessor: BasePreprocessor) -> "PreprocessingPipeline":
        """
        Add a preprocessing step.

        Args:
            name: Name for this step
            preprocessor: Preprocessor instance

        Returns:
            Self for chaining
        """
        self.steps.append((name, preprocessor))
        logger.info(f"Added step '{name}': {preprocessor.__class__.__name__}")
        return self

    def add_preprocessor(
        self,
        method: str,
        name: Optional[str] = None,
        **kwargs
    ) -> "PreprocessingPipeline":
        """
        Add a preprocessor by method name.

        Args:
            method: Preprocessor method name
            name: Optional step name (defaults to method name)
            **kwargs: Preprocessor configuration

        Returns:
            Self for chaining
        """
        preprocessor = PreprocessorFactory.create(method, **kwargs)
        step_name = name or method
        return self.add_step(step_name, preprocessor)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "PreprocessingPipeline":
        """
        Fit all preprocessors in sequence.

        Args:
            X: Feature data
            y: Target data (for supervised preprocessors)
            feature_names: Optional feature names

        Returns:
            Self
        """
        logger.info(f"Fitting pipeline with {len(self.steps)} steps")

        if feature_names is not None:
            self._feature_names_in = feature_names
        else:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        X_transformed = X.copy()
        current_feature_names = self._feature_names_in.copy()

        for step_name, preprocessor in self.steps:
            logger.info(f"Fitting step '{step_name}'...")

            # Set feature names
            preprocessor.set_feature_names(current_feature_names)

            # Fit preprocessor
            preprocessor.fit(X_transformed, y)

            # Transform data for next step
            X_transformed = preprocessor.transform(X_transformed)

            # Update feature names
            out_names = preprocessor.get_feature_names_out()
            if out_names:
                current_feature_names = out_names

            logger.info(f"Step '{step_name}' complete. Shape: {X_transformed.shape}")

        self._feature_names_out = current_feature_names
        self._fitted = True

        logger.info(f"Pipeline fitted. Input: {X.shape} -> Output: {X_transformed.shape}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data through all fitted preprocessors.

        Args:
            X: Feature data to transform

        Returns:
            Transformed data
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        X_transformed = X.copy()

        for step_name, preprocessor in self.steps:
            X_transformed = preprocessor.transform(X_transformed)

        return X_transformed

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit and transform data.

        Args:
            X: Feature data
            y: Target data
            feature_names: Optional feature names

        Returns:
            Transformed data
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data through all preprocessors (in reverse).

        Args:
            X: Transformed data

        Returns:
            Original data (approximately)

        Note:
            Not all preprocessors support inverse_transform.
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before inverse_transform")

        X_inverse = X.copy()

        # Apply inverse transforms in reverse order
        for step_name, preprocessor in reversed(self.steps):
            try:
                X_inverse = preprocessor.inverse_transform(X_inverse)
            except NotImplementedError:
                logger.warning(
                    f"Step '{step_name}' does not support inverse_transform"
                )
                break

        return X_inverse

    def get_step(self, name: str) -> Optional[BasePreprocessor]:
        """
        Get a preprocessor by step name.

        Args:
            name: Step name

        Returns:
            Preprocessor or None
        """
        for step_name, preprocessor in self.steps:
            if step_name == name:
                return preprocessor
        return None

    def get_feature_names_out(self) -> Optional[List[str]]:
        """Get output feature names."""
        return self._feature_names_out

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            "n_steps": len(self.steps),
            "steps": [
                {"name": name, "type": p.__class__.__name__, "params": p.get_params()}
                for name, p in self.steps
            ],
            "is_fitted": self._fitted,
            "n_features_in": len(self._feature_names_in) if self._feature_names_in else None,
            "n_features_out": len(self._feature_names_out) if self._feature_names_out else None
        }

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "n_steps": len(self.steps),
            "steps": [
                {"name": name, "type": p.__class__.__name__}
                for name, p in self.steps
            ],
            "is_fitted": self._fitted
        }

    def save(self, save_path: Union[str, Path]) -> Path:
        """
        Save pipeline to disk.

        Args:
            save_path: Path to save pipeline

        Returns:
            Path where pipeline was saved
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline_data = {
            "steps": self.steps,
            "fitted": self._fitted,
            "feature_names_in": self._feature_names_in,
            "feature_names_out": self._feature_names_out
        }

        with open(save_path, 'wb') as f:
            pickle.dump(pipeline_data, f)

        logger.info(f"Saved pipeline to: {save_path}")
        return save_path

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> "PreprocessingPipeline":
        """
        Load pipeline from disk.

        Args:
            load_path: Path to saved pipeline

        Returns:
            Loaded pipeline instance
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {load_path}")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        pipeline = cls(steps=data["steps"])
        pipeline._fitted = data["fitted"]
        pipeline._feature_names_in = data["feature_names_in"]
        pipeline._feature_names_out = data["feature_names_out"]

        logger.info(f"Loaded pipeline from: {load_path}")
        return pipeline

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PreprocessingPipeline":
        """
        Create pipeline from configuration dictionary.

        Args:
            config: Configuration with 'steps' key

        Returns:
            Pipeline instance
        """
        pipeline = cls()

        steps_config = config.get("steps", [])
        for step in steps_config:
            method = step.get("method")
            name = step.get("name", method)
            params = step.get("params", {})

            if method:
                pipeline.add_preprocessor(method, name=name, **params)

        return pipeline

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        step_names = [name for name, _ in self.steps]
        return f"PreprocessingPipeline(steps={step_names}, fitted={self._fitted})"
