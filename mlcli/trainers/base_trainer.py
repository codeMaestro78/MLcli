"""
Base Trainer Abstract Class

Defines the interface that all model trainers must implement,
ensuring consistent behavior across ML and DL frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers.

    All trainer implementations must inherit from this class and implement
    the required abstract methods: train, evaluate, save, load.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.training_history: Dict[str, Any] = {}

        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on provided data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Dictionary containing training history/metrics
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (if supported).

        Args:
            X: Input features

        Returns:
            Predicted probabilities or None if not supported
        """
        pass

    @abstractmethod
    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save model to disk in specified formats.

        Args:
            save_dir: Directory to save model files
            formats: List of formats to save in (e.g., ['pickle', 'onnx'])

        Returns:
            Dictionary mapping format names to file paths
        """
        pass

    @abstractmethod
    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to model file
            model_format: Format of the model file
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default hyperparameters for this model.

        Returns:
            Dictionary of default parameters
        """
        pass

    def get_model(self) -> Any:
        """
        Get the underlying model object.

        Returns:
            Model object

        Raises:
            RuntimeError: If model hasn't been trained or loaded
        """
        if self.model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no trained model. " "Call train() or load() first."
            )
        return self.model

    def get_config(self) -> Dict[str, Any]:
        """
        Get trainer configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history.

        Returns:
            Training history dictionary
        """
        return self.training_history.copy()

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(status={status})"
