"""
LightGBM Trainer

LightGBM-based trainer for gradient boosting classification and regression.
LightGBM is a fast, distributed, high-performance gradient boosting framework.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
import lightgbm as lgb
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="lightgbm",
    description="LightGBM gradient boosting classifier/regressor",
    framework="lightgbm",
    model_type="classification",
)
class LightGBMTrainer(BaseTrainer):
    """
    Trainer for LightGBM models.

    LightGBM is a gradient boosting framework that uses tree-based learning
    algorithms. It is designed to be distributed and efficient with:
    - Faster training speed and higher efficiency
    - Lower memory usage
    - Better accuracy
    - Support for parallel and GPU learning
    - Capable of handling large-scale data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Task type: classification or regression
        self.task_type = self.config.get("task_type", "classification")

        # Early stopping configuration
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 50)

        logger.info(
            f"Initialized LightGBMTrainer with n_estimators={self.model_params['n_estimators']}, "
            f"task_type={self.task_type}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """
        logger.info(f"Training LightGBM on {X_train.shape[0]} samples")

        # Choose classifier or regressor based on task type
        if self.task_type == "regression":
            self.model = lgb.LGBMRegressor(**self.model_params)
        else:
            self.model = lgb.LGBMClassifier(**self.model_params)

        # Prepare callbacks for early stopping
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),  # Suppress verbose output
            ]

        # Train model
        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train)

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)

        if self.task_type == "classification":
            y_train_proba = self.model.predict_proba(X_train)
            train_metrics = compute_metrics(
                y_train, y_train_pred, y_train_proba, task="classification"
            )
        else:
            train_metrics = compute_metrics(y_train, y_train_pred, task="regression")

        # Feature importance
        feature_importance = self.model.feature_importances_.tolist()

        self.training_history = {
            "train_metrics": train_metrics,
            "feature_importance": feature_importance,
            "n_features": X_train.shape[1],
            "best_iteration": getattr(self.model, "best_iteration_", None),
            "task_type": self.task_type,
        }

        if self.task_type == "classification":
            self.training_history["n_classes"] = len(np.unique(y_train))

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True

        metric_key = "accuracy" if self.task_type == "classification" else "r2"
        logger.info(f"Training complete. {metric_key}: {train_metrics.get(metric_key, 0):.4f}")

        if self.training_history.get("best_iteration"):
            logger.info(f"Best iteration: {self.training_history['best_iteration']}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate LightGBM model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)

        if self.task_type == "classification":
            y_proba = self.model.predict_proba(X_test)
            metrics = compute_metrics(y_test, y_pred, y_proba, task="classification")
            logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        else:
            metrics = compute_metrics(y_test, y_pred, task="regression")
            logger.info(f"Evaluation complete. R2: {metrics['r2']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values.

        Args:
            X: Input features

        Returns:
            Predicted labels/values
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (classification only).

        Args:
            X: Input features

        Returns:
            Predicted probabilities or None for regression
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if self.task_type == "regression":
            return None

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance scores.

        Returns:
            Dictionary with different importance types
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return {
            "split": self.model.booster_.feature_importance(importance_type="split"),
            "gain": self.model.booster_.feature_importance(importance_type="gain"),
        }

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save LightGBM model.

        Args:
            save_dir: Directory to save models
            formats: List of formats ('pickle', 'joblib', 'lightgbm', 'onnx')

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
                path = save_dir / "lightgbm_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "model": self.model,
                            "config": self.config,
                            "task_type": self.task_type,
                        },
                        f,
                    )
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "lightgbm_model.joblib"
                joblib.dump(
                    {
                        "model": self.model,
                        "config": self.config,
                        "task_type": self.task_type,
                    },
                    path,
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            elif fmt == "lightgbm":
                # Native LightGBM format
                path = save_dir / "lightgbm_model.txt"
                self.model.booster_.save_model(str(path))
                saved_paths["lightgbm"] = path
                logger.info(f"Saved LightGBM model to {path}")

            elif fmt == "onnx":
                path = save_dir / "lightgbm_model.onnx"
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
        Load LightGBM model.

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
                self.task_type = data.get("task_type", "classification")

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.config = data.get("config", {})
            self.task_type = data.get("task_type", "classification")

        elif model_format == "lightgbm":
            # Load native LightGBM format
            booster = lgb.Booster(model_file=str(model_path))
            # Wrap in sklearn API
            if self.task_type == "regression":
                self.model = lgb.LGBMRegressor()
            else:
                self.model = lgb.LGBMClassifier()
            self.model._Booster = booster

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
            "max_depth": -1,  # No limit
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "n_jobs": -1,
            "boosting_type": "gbdt",  # 'gbdt', 'dart', 'goss', 'rf'
            "verbose": -1,
        }

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameter ranges
        """
        return {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [-1, 5, 10, 15, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [10, 20, 30, 50],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        }
