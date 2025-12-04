"""
Support Vector Machine Trainer

Sklearn-based trainer for SVM classification.
"""

import joblib
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging
from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="svm",
    description="Support Vector Machine with RBF/Linear/Poly kernels",
    framework="sklearn",
    model_type="classification",
)
class SVMTrainer(BaseTrainer):
    """
    Trainer for Support Vector Machine models.

    Supports multiple kernels (RBF, linear, polynomial) with
    automatic feature scaling and probability estimation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SVM trainer.

        Args:
            config: Configuration dictionary with model parameters
        """

        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Features scaling (highly recommended for SVM)
        self.scale_features = self.config.get("scale_features", True)
        self.scaler = StandardScaler() if self.scale_features else None

        logger.info(f"Initialized SVMTrainer with kernel={self.model_params['kernel']}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train SVM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """

        logger.info(f"Training SVM on {X_train.shape[0]} samples")

        # Scale features
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
            logger.debug("Applied feature scaling")

        # Train model
        self.model = SVC(**self.model_params)
        self.model.fit(X_train, y_train)

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = None

        if self.model_params.get("probability", False):
            y_train_proba = self.model.predict_proba(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, task="classification")

        self.training_history = {
            "train_metrics": train_metrics,
            "n_support_vectors": self.model.n_support_.tolist(),
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            test_metrics = self.evaluate(X_val, y_val)
            self.training_history["test_metrics"] = test_metrics

        self.is_trained = True
        logger.info(f"Training complete. Accuracy: {train_metrics["accuracy"]:.4f}")
        logger.info(f"Support vectors: {sum(self.model.n_support_)}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate SVM model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if self.scale_features and self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        y_pred = self.model.predict(X_test)
        y_proba = None

        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba, task="classification")

        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")

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
            raise RuntimeError("Model not trained, Call train() first.")

        if self.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities or None
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if not hasattr(self.model, "predict_proba"):
            logger.warning("SVM was not trained with probability=True")
            return None

        if self.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save SVM model.

        Args:
            save_dir: Directory to save models
            formats: List of formats

        Returns:
            Dictionary of saved paths
        """

        if self.model is None:
            raise RuntimeError("No model to save. Train model first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for fmt in formats:
            if fmt == "pickle":
                path = save_dir / "svm_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {"model": self.model, "scaler": self.scaler, "config": "self.config"}, f
                    )

                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "svm_model.joblib"
                joblib.dump(
                    {"model": self.model, "scaler": self.scaler, "config": self.config}, path
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            elif fmt == "onnx":
                path = save_dir / "svm_model.onnx"
                try:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType

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
                logger.warning(f"Unsupported format: {fmt}")

        return saved_paths

    def load(self, model_name: Path, model_format: str) -> None:
        """
        Load SVM model.

        Args:
            model_path: Path to model file
            model_format: Format of the model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

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

        else:
            raise ValueError(f"Unsupported format: {model_format}")

        self.is_trained = True
        logger.info(f"Loaded {model_format} model from {model_path}")

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default hyperparameters.

        Returns:
            Default parameters
        """
        return {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        }
