"""
Random Forest Trainer

Sklearn-based trainer for Random Forest classification.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="random_forest",
    description="Random Forest ensemble classifier method",
    framework="sklearn",
    model_type="classification",
)
class RFTrainer(BaseTrainer):
    """
    Trainer for Random Forest models.

    Ensemble learning method using multiple decision trees
    with feature randomness and bootstrap aggregation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        logger.info(f"Initialized RFTrainer with n_estimators={self.model_params['n_estimators']}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """
        logger.info(f"Training Random Forest on {X_train.shape[0]} samples")

        # Train model
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, task="classification")

        # Feature importance
        feature_importance = self.model.feature_importances_.tolist()

        self.training_history = {
            "train_metrics": train_metrics,
            "feature_importance": feature_importance,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
            "oob_score": self.model.oob_score_ if self.model_params.get("oob_score") else None,
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True
        logger.info(f"Training complete. Accuracy: {train_metrics['accuracy']:.4f}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate Random Forest model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
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
            raise RuntimeError("Model not trained. Call train() first.")

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

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of feature importance values
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.feature_importances_

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save Random Forest model.

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
                path = save_dir / "rf_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump({"model": self.model, "config": self.config}, f)
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "rf_model.joblib"
                joblib.dump({"model": self.model, "config": self.config}, path)
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            elif fmt == "onnx":
                path = save_dir / "rf_model.onnx"
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

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load Random Forest model.

        Args:
            model_path: Path to model file
            model_format: Format of the model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_format == "pickle":
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.config = data.get("config", {})

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
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
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }
