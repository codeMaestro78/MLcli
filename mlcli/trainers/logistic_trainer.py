"""
Logistic Regression Trainer

Sklearn-based trainer for logistic regression classification.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="logistic_regression",
    description="Logistic Regression classifier with L2 regularization",
    framework="sklearn",
    model_type="classification",
)
class LogisticRegressionTrainer(BaseTrainer):
    """
    Trainer for Logistic Regression models.

    Supports automatic feature scaling, multiple solvers, and
    regularization strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Logistic Regression trainer.

        Args:
            config: Configuration dictionary with model parameters
        """

        super().__init__(config)

        # Extract parameters from config
        params = self.config.get("params", {})

        # Merge with defaults
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Feature scaling option
        self.scale_features = self.config.get("scale_features", True)
        self.scaler = StandardScaler() if self.scale_features else None

        logger.info(f"Initialized LogisticRegressionTrainer with params: {self.model_params}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history with metrics
        """
        logger.info(f"Training Logistic Regression on {X_train.shape[0]} samples")

        # Scale features if enabled
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
            logger.debug("Applied feature scaling")

        # Create and train model
        self.model = LogisticRegression(**self.model_params)
        self.model.fit(X_train, y_train)

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, task="classification")

        self.training_history = {
            "train_metrics": train_metrics,
            "n_iterations": self.model.n_iter_[0] if hasattr(self.model, "n_iter_") else None,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
        }

        # Validation metircs if provided
        if X_val is not None and y_val is not None:
            test_metics = self.evaluate(X_val, y_val)
            self.training_history["test_metrics"] = test_metics

        self.is_trained = True
        logger.info(f"Training complete. Accuracy: {train_metrics['accuracy']:.4f}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """

        if self.model is None:
            raise RuntimeError(f"Model not trained. Call train() first.")

        # Scale features if enabled
        if self.scale_features and self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba, task="classification")

        logger.info(f"Evaluation complete. Accuracy :{metrics['accuracy']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first. ")

        if self.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if self.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save model in specified formats.

        Args:
            save_dir: Directory to save models
            formats: List of formats ('pickle', 'joblib', 'onnx')

        Returns:
            Dictionary mapping format to file path
        """

        if self.model is None:
            raise RuntimeError("No model to save. Train model first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for fmt in formats:
            if fmt == "pickle":
                path = save_dir / "logistic_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {"model": self.model, "scaler": self.scaler, "config": self.config}, f
                    )
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "logistic_model.joblib"
                joblib.dump(
                    {"model": self.model, "scaler": self.scaler, "config": self.config}, path
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            elif fmt == "onnx":
                path = save_dir / "logistic_model.onnx"
                try:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType

                    # Determine input shape
                    n_features = self.training_history.get("n_features", 1)
                    initial_type = [("float_input", FloatTensorType([None, n_features]))]

                    onx = convert_sklearn(self.model, initial_types=initial_type)

                    with open(path, "wb") as f:
                        f.write(onx.SerializeToString())

                    saved_paths["onnx"] = path
                    logger.info(f"Saved ONNX model to {path}")

                except Exception as e:
                    logger.error(f"Failed to save ONNX model: {e}")

            else:
                logger.error(f"Unsupported format: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to model file
            model_format: Format ('pickle', 'joblib', 'onnx')
        """

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_format == "pickle":
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data.get("scaler")
                self.config = data.get("config", {})

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.scaler = data.get("scaler")
            self.config = data.get("config", {})

        elif model_format == "onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(str(model_path))
            logger.warning("ONNX models have limited functionality")

        else:
            raise ValueError(f"Unsupported format: {model_format}")

        self.is_trained = True
        logger.info(f"Loaded {model_format} model from {model_path}")

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default hyperparameters.

        Returns:
            Default parameters dictionary
        """
        return {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        }
