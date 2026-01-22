"""
Random Forest Trainer

Sklearn-based trainer for Random Forest classification.
"""

from __future__ import annotations

import json
import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import sklearn

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class RFConfig(BaseModel):
    n_estimators: int = Field(100, ge=1)
    max_depth: Optional[int] = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_features: str = "sqrt"
    bootstrap: bool = True
    oob_score: bool = False
    class_weight: Optional[str] = None
    random_state: int = 42
    n_jobs: int = -1
    warm_start: bool = False


@register_model(
    name="random_forest",
    description="Random Forest ensemble classifier",
    framework="sklearn",
    model_type="classification",
    supports_multiclass=True,
    supports_onnx=True,
    supports_probabilities=True,
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

        try:
            params = self.config.get("params", {})
            self.model_params = RFConfig(**params).dict()
        except ValidationError as e:
            raise ValueError(f"Invalid RandomForest config: {e}")

        if self.model_params["oob_score"] and not self.model_params["bootstrap"]:
            raise ValueError("oob_score=True requires bootstrap=True")

        self.model: Optional[RandomForestClassifier] = None
        self.backend: str = "sklearn"

        logger.info(
            "Initialized RFTrainer", extra={"params": json.dumps(self.model_params, sort_keys=True)}
        )

    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y length mismatch")

    def _check_is_trained(self):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not found")

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
        self._validate_inputs(X_train, y_train)
        logger.info(
            "Starting Random Forest training",
            extra={
                "samples": X_train.shape[0],
                "features": X_train.shape[1],
            },
        )

        # Train model
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, task="classification")

        # OOB score safety
        oob_score = (
            getattr(self.model, "oob_score_", None) if self.model_params["oob_score"] else None
        )

        self.training_history = {
            "train_metrics": train_metrics,
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
            "feature_importance": self.model.feature_importances_.tolist(),
            "oob_score": oob_score,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        }

        self.is_trained = True

        # Validation metrics
        if X_val is not None and y_val is not None:
            self._validate_inputs(X_val, y_val)
            self.training_history["val_metrics"] = self.evaluate(X_val, y_val)

        logger.info(
            "Training completed",
            extra={"accuracy": train_metrics.get("accuracy")},
        )

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
        self._check_is_trained()
        self._validate_inputs(X_test, y_test)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba, task="classification")

        logger.info(
            "Evaluation completed",
            extra={"accuracy": metrics.get("accuracy")},
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        self._check_is_trained()
        self._validate_inputs(X)

        if self.backend == "sklearn":
            return self.model.predict(X)

        raise RuntimeError("Predict_proba not supported for this backend")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        self._check_is_trained()
        self._validate_inputs(X)

        if self.backend == "sklearn":
            return self.model.predict_proba(X)

        raise RuntimeError("Predict_proba not supported for this backend")

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of feature importance values
        """
        self._check_is_trained()

        return self.model.feature_importances_

    def get_permutation_importance(self, X: np.ndarray, y: np.ndarray, n_repeats: int = 5):
        self._check_is_trained()
        return permutation_importance(self.model, X, y, n_repeats=n_repeats, n_jobs=-1)

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save Random Forest model.

        Args:
            save_dir: Directory to save models
            formats: List of formats

        Returns:
            Dictionary of saved paths
        """
        self._check_is_trained()

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
            self.backend = "sklearn"

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.config = data.get("config", {})
            self.backend = "sklearn"

        else:
            raise ValueError(f"Unsupported format : {model_format}")

        self.is_trained = True
        logger.info(f"Loaded {model_format} model from {model_path}")

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default hyperparameters.

        Returns:
            Default parameters
        """
        return RFConfig().dict()
