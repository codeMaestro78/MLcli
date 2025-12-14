"""
One-Class SVM Trainer

One-Class SVM for anomaly/outlier detection.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    name="one_class_svm",
    description="One-Class SVM anomaly detection algorithm",
    framework="sklearn",
    model_type="anomaly_detection",
)
class OneClassSVMTrainer(BaseTrainer):
    """
    Trainer for One-Class SVM anomaly detection.

    One-Class SVM learns a decision boundary around normal data.
    Points outside this boundary are considered anomalies.

    Features:
    - Effective for high-dimensional data
    - Works well when anomalies are rare
    - RBF kernel captures complex boundaries
    - No assumption about data distribution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize One-Class SVM trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        logger.info(
            f"Initialized OneClassSVMTrainer with kernel={self.model_params['kernel']}, "
            f"nu={self.model_params['nu']}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,  # Can be used for validation if available
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train One-Class SVM model.

        Args:
            X_train: Training features (assumed mostly normal data)
            y_train: Optional labels (1 for normal, -1 for anomaly) for evaluation
            X_val: Validation features (optional)
            y_val: Validation labels (optional, for evaluation)

        Returns:
            Training history with anomaly detection metrics
        """
        logger.info(
            f"Training One-Class SVM on {X_train.shape[0]} samples with "
            f"kernel={self.model_params['kernel']}, nu={self.model_params['nu']}"
        )

        # Create and fit model
        self.model = OneClassSVM(**self.model_params)
        self.model.fit(X_train)

        # Get predictions and anomaly scores
        predictions = self.model.predict(X_train)
        scores = self.model.decision_function(X_train)

        # Count anomalies
        n_anomalies = (predictions == -1).sum()
        anomaly_ratio = n_anomalies / len(predictions)

        # Base training metrics
        train_metrics = {
            "n_anomalies": int(n_anomalies),
            "n_normal": int((predictions == 1).sum()),
            "anomaly_ratio": float(anomaly_ratio),
            "mean_anomaly_score": float(scores.mean()),
            "std_anomaly_score": float(scores.std()),
            "min_anomaly_score": float(scores.min()),
            "max_anomaly_score": float(scores.max()),
            "n_support_vectors": int(self.model.n_support_),
        }

        # If ground truth labels are provided, compute classification metrics
        if y_train is not None:
            train_metrics.update(self._compute_detection_metrics(y_train, predictions, scores))

        self.training_history = {
            "train_metrics": train_metrics,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
            "nu": self.model_params["nu"],
            "kernel": self.model_params["kernel"],
        }

        # Validation metrics
        if X_val is not None:
            val_predictions = self.model.predict(X_val)
            val_scores = self.model.decision_function(X_val)

            val_metrics = {
                "n_anomalies": int((val_predictions == -1).sum()),
                "anomaly_ratio": float((val_predictions == -1).sum() / len(val_predictions)),
            }

            if y_val is not None:
                val_metrics.update(
                    self._compute_detection_metrics(y_val, val_predictions, val_scores)
                )

            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True

        logger.info(
            f"Training complete. Detected {n_anomalies} anomalies "
            f"({anomaly_ratio:.2%} of data). "
            f"Support vectors: {self.model.n_support_}"
        )

        return self.training_history

    def _compute_detection_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute anomaly detection metrics when ground truth is available.

        Args:
            y_true: True labels (1 for normal, -1 for anomaly)
            y_pred: Predicted labels
            scores: Anomaly scores

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Convert to binary: 1 for anomaly, 0 for normal
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)

        metrics["precision"] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        metrics["recall"] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        metrics["f1"] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

        # ROC AUC (using anomaly scores, inverted since lower = more anomalous)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, -scores))
        except ValueError:
            metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        if cm.shape == (2, 2):
            metrics["true_negatives"] = int(cm[0, 0])
            metrics["false_positives"] = int(cm[0, 1])
            metrics["false_negatives"] = int(cm[1, 0])
            metrics["true_positives"] = int(cm[1, 1])

        return metrics

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Evaluate anomaly detection on test data.

        Args:
            X_test: Test features
            y_test: Optional ground truth labels

        Returns:
            Detection metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        predictions = self.model.predict(X_test)
        scores = self.model.decision_function(X_test)

        n_anomalies = (predictions == -1).sum()

        metrics = {
            "n_anomalies": int(n_anomalies),
            "anomaly_ratio": float(n_anomalies / len(predictions)),
            "mean_anomaly_score": float(scores.mean()),
        }

        if y_test is not None:
            metrics.update(self._compute_detection_metrics(y_test, predictions, scores))
            logger.info(f"Evaluation complete. F1: {metrics.get('f1', 'N/A')}")
        else:
            logger.info(f"Evaluation complete. Detected {n_anomalies} anomalies")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if samples are anomalies.

        Args:
            X: Input features

        Returns:
            Predicted labels: 1 for inliers, -1 for outliers
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get anomaly scores (not probabilities, but can be used similarly).

        Args:
            X: Input features

        Returns:
            Anomaly scores normalized to [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Return scores normalized to [0, 1] range
        scores = self.model.decision_function(X)
        # Normalize: anomalies have negative scores, normal have positive
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return np.column_stack([1 - normalized, normalized])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw anomaly scores.

        Args:
            X: Input features

        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.decision_function(X)

    def get_anomalies(self, X: np.ndarray, return_scores: bool = False) -> np.ndarray:
        """
        Get indices of detected anomalies.

        Args:
            X: Input features
            return_scores: Whether to also return anomaly scores

        Returns:
            Indices of anomalies (and optionally their scores)
        """
        predictions = self.predict(X)
        anomaly_indices = np.where(predictions == -1)[0]

        if return_scores:
            scores = self.decision_function(X)
            return anomaly_indices, scores[anomaly_indices]

        return anomaly_indices

    def get_support_vectors(self) -> np.ndarray:
        """
        Get support vectors from trained model.

        Returns:
            Array of support vectors
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.support_vectors_

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save One-Class SVM model.

        Args:
            save_dir: Directory to save models
            formats: List of formats ('pickle', 'joblib')

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
                path = save_dir / "one_class_svm_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "model": self.model,
                            "config": self.config,
                            "training_history": self.training_history,
                        },
                        f,
                    )
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "one_class_svm_model.joblib"
                joblib.dump(
                    {
                        "model": self.model,
                        "config": self.config,
                        "training_history": self.training_history,
                    },
                    path,
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            else:
                logger.warning(f"Unsupported format for One-Class SVM: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load One-Class SVM model.

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
                self.training_history = data.get("training_history", {})

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.config = data.get("config", {})
            self.training_history = data.get("training_history", {})

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
            "gamma": "scale",
            "nu": 0.1,  # Upper bound on fraction of training errors
            "tol": 1e-3,
            "shrinking": True,
            "cache_size": 200,
            "max_iter": -1,  # No limit
        }

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameter ranges
        """
        return {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "nu": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            "degree": [2, 3, 4],  # For poly kernel
            "coef0": [0.0, 0.5, 1.0],  # For poly/sigmoid kernels
        }
